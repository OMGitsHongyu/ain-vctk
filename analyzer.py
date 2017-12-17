import sys
try:
    sys.path.remove('/home/hongyuz/.local/lib/python2.7/site-packages')
    sys.path.append('/home/hongyuz/.local/lib/python2.7/site-packages')
except ValueError:
    pass
import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import librosa
import pyworld as pw
import tensorflow as tf

FFT_SIZE = 1024
EPSILON = 1e-10
SP_DIM = 513
FEAT_DIM = SP_DIM + SP_DIM + 1 + 1 + 1  # [sp, ap, f0, en, s]

def wav2pw(x, fs=16000, fft_size=FFT_SIZE):
    ''' Extract WORLD feature from waveform '''
    _f0, t = pw.dio(x, fs, f0_ceil=500)            # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  		   # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(x, f0, t, fs, fft_size=fft_size)   # extract aperiodicity
    return {
        'f0': f0,
        'sp': sp,
        'ap': ap,
    }

def pw2wav(features, feat_dim=513, fs=16000):
    ''' NOTE: Use `order='C'` to ensure Cython compatibility '''
    if isinstance(features, dict):
        en = np.reshape(features['en'], [-1, 1])
        sp = np.power(10., features['sp'])
        sp = en * sp
        return pw.synthesize(
            features['f0'].astype(np.float64).copy(order='C'),
            sp.astype(np.float64).copy(order='C'),
            features['ap'].astype(np.float64).copy(order='C'),
            fs,
        )
    features = features.astype(np.float64)
    sp = features[:, :feat_dim]
    ap = features[:, feat_dim:feat_dim*2]
    f0 = features[:, feat_dim*2]
    en = features[:, feat_dim*2 + 1]
    en = np.reshape(en, [-1, 1])
    sp = np.power(10., sp)
    sp = en * sp
    return pw.synthesize(
        f0.copy(order='C'),
        sp.copy(order='C'),
        ap.copy(order='C'),
        fs
    )

def extract(wavfile, fft_size=FFT_SIZE, dtype=np.float32):
    ''' Basic (WORLD) feature extraction '''
    if isinstance(wavfile, str):
    	x, _ = librosa.load(wavfile, sr=16000, mono=True, dtype=np.float64)
    elif isinstance(wavfile, np.ndarray):
	x = wavfile
    else:
	raise ValueError('input wavfile must be either filename or numpy.ndarray')
    features = wav2pw(x, 16000, fft_size=fft_size)
    ap = features['ap']
    f0 = features['f0'].reshape(-1,1)
    sp = features['sp']
    en = np.sum(sp + EPSILON, axis=1, keepdims=True)
    sp = np.log10(sp / en)
    return np.concatenate([sp, ap, f0, en], axis=1).astype(dtype)

def convert_f0(f0, src, trg):
    if isinstance(src, (list, tuple)):
        mu_s, std_s = src
    elif isinstance(src, (float, np.float32, int, np.int32, np.int64, str)):
        mu_s, std_s = np.fromfile(os.path.join('data/225/etc', '{}.npf'.format(int(src))), np.float32)
    else:
        raise TypeError('src must be numeric, string or tuple/list, got {}'.format(type(src)))
    if isinstance(trg, (list, tuple)):
        mu_t, std_t = trg
    elif isinstance(trg, (float, np.float32, int, np.int32, np.int64, str)):
        mu_t, std_t = np.fromfile(os.path.join('data/225/etc', '{}.npf'.format(int(trg))), np.float32)
    else:
        raise TypeError('trg must be numeric string or tuple/list, got {}'.format(type(trg)))
    lf0 = np.where(f0 > 1., np.log(f0), f0)
    lf0 = np.where(lf0 > 1., (lf0 - mu_s)/std_s * std_t + mu_t, lf0)
    lf0 = np.where(lf0 > 1., np.exp(lf0), lf0)
    return lf0

def convert_feature(spec, other_feature, src, trg, normalizer=None):
    if normalizer:
        spec = normalizer.backward_process(spec)
    f0 = convert_f0(other_feature[:,-3], src, trg)
    return np.hstack((spec, other_feature[:,:-3], f0.reshape(-1,1), other_feature[:,-2].reshape(-1,1)))

#------------------------save audios to binary files-----------------------------
def find_paired_files(dir_to_source, target, sources=range(225,251), test=0.25):
    '''Finds all training (data and label) files matching the pattern.'''
    label_files = glob.glob(dir_to_source+'/txt/'+'p'+str(target)+'/*')
    text_label = {}
    for label_file in label_files:
        with open(label_file, 'r') as fl:
            text_label[fl.read().lstrip().rstrip()] = dir_to_source+'/wav48/'+'p'+str(target)+'/'+label_file.split('/')[-1][:-4]+'.wav'

    train_idx, test_idx = train_test_split(text_label.items(), test_size=test)
    train_label, test_label = dict(train_idx), dict(test_idx)
    train_wav, test_wav = [], []
    for insider in [i for i in sources if i != target]:
        train_files = glob.glob(dir_to_source+'/txt/'+'p'+str(insider)+'/*')
        for train_file in train_files:
            with open(train_file, 'r') as ft:
                text = ft.read().lstrip().rstrip()
                if text in train_label:
                    train_audio = dir_to_source+'/wav48/'+'p'+str(insider)+'/'+train_file.split('/')[-1][:-4]+'.wav'
                    train_wav.append((train_audio, text_label[text]))
                if text in test_label:
                    train_audio = dir_to_source+'/wav48/'+'p'+str(insider)+'/'+train_file.split('/')[-1][:-4]+'.wav'
                    test_wav.append((train_audio, text_label[text]))
    return train_wav, test_wav


def extract_and_save_bin_to(dir_to_bin, dir_to_source, target, top_db=20, sources=range(225,251), test=0.25):
    train_pairs, test_pairs = find_paired_files(dir_to_source, target, sources=sources, test=test)
    print('{} wavs in training set and {} wavs in test set'.format(len(train_pairs), len(test_pairs)))
    label_set = set()
    train_set, test_set = [], []
    if os.path.exists(dir_to_bin):
        os.system('rm -rf '+dir_to_bin)
    os.makedirs(dir_to_bin)
    os.makedirs(dir_to_bin+'/train/')
    os.makedirs(dir_to_bin+'/test/')
    os.makedirs(dir_to_bin+'/label/')
    os.makedirs(dir_to_bin+'/etc/')
    for wav_pair in train_pairs:
        output_filename = os.path.splitext(wav_pair[0])[0].split('/')[-1]
	label_filename = os.path.splitext(wav_pair[1])[0].split('/')[-1]
        label_set.add(wav_pair[1])
        output_file = os.path.join(dir_to_bin+'/train/', '{}.bin'.format(output_filename))
	label_file = os.path.join(dir_to_bin+'/label/', '{}.bin'.format(label_filename))
	train_set.append((output_file, label_file))
        if os.path.exists(output_file):
            print('{} already exists'.format(output_file))
        else:
            print('Process {}'.format(output_file))
            wave, _ = librosa.load(wav_pair[0], sr=16000, dtype=np.float64)
            wave_trim, _ = librosa.effects.trim(wave, top_db=top_db)
            features = extract(wave_trim)
            speaker = int(output_filename.split('_')[0][1:]) * np.ones((features.shape[0],1),np.float32)
            features = np.concatenate([features, speaker], 1)
            print('features', features.shape)
            with open(output_file, 'wb') as fp:
                fp.write(features.tostring())
    for wav_pair in test_pairs:
        output_filename = os.path.splitext(wav_pair[0])[0].split('/')[-1]
	label_filename = os.path.splitext(wav_pair[1])[0].split('/')[-1]
        label_set.add(wav_pair[1])
        output_file = os.path.join(dir_to_bin+'/test/', '{}.bin'.format(output_filename))
	label_file = os.path.join(dir_to_bin+'/label/', '{}.bin'.format(label_filename))
	test_set.append((output_file, label_file))
        if os.path.exists(output_file):
            print('{} already exists'.format(output_file))
        else:
            print('Process {}'.format(output_file))
            wave, _ = librosa.load(wav_pair[0], sr=16000, dtype=np.float64)
            wave_trim, _ = librosa.effects.trim(wave, top_db=top_db)
            features = extract(wave_trim)
            speaker = int(output_filename.split('_')[0][1:]) * np.ones((features.shape[0],1),np.float32)
            features = np.concatenate([features, speaker], 1)
            print('features', features.shape)
            with open(output_file, 'wb') as fp:
                fp.write(features.tostring())
    print('{} wavs in label set'.format(len(label_set)))
    for label_file in label_set:
        output_filename = os.path.splitext(label_file)[0].split('/')[-1]
        output_file = os.path.join(dir_to_bin+'/label/', '{}.bin'.format(output_filename))
        if os.path.exists(output_file):
            print('{} already exists'.format(output_file))
        else:
            print('Process {}'.format(output_file))
            wave, _ = librosa.load(label_file, sr=16000, dtype=np.float64)
            wave_trim, _ = librosa.effects.trim(wave, top_db=top_db)
            features = extract(wave_trim)
            speaker = int(output_filename.split('_')[0][1:]) * np.ones((features.shape[0],1),np.float32)
            features = np.concatenate([features, speaker], 1)
            print('features', features.shape)
            with open(output_file, 'wb') as fp:
                fp.write(features.tostring())

    with open(dir_to_bin+'/etc/train_map.txt', 'w') as fp:
        for filename_pair in train_set:
            fp.write(filename_pair[0]+','+filename_pair[1]+'\n')

    with open(dir_to_bin+'/etc/test_map.txt', 'w') as fp:
        for filename_pair in test_set:
            fp.write(filename_pair[0]+','+filename_pair[1]+'\n')

def save_pair_to(dir_to_pair, map_txt):
    pair_map = []
    with open(map_txt) as fp:
        for line in fp.readlines():
            pair_map.append(line.rstrip().split(','))
    print('{} wavs in the set '.format(len(pair_map)))
    if os.path.exists(dir_to_pair):
        os.system('rm -rf '+dir_to_pair)
    os.system('mkdir -p '+dir_to_pair)
    for wav_pair in pair_map:
        output_filename = os.path.splitext(wav_pair[0])[0].split('/')[-1]+'-'+\
                          os.path.splitext(wav_pair[1])[0].split('/')[-1]
        output_file = os.path.join(dir_to_pair, '{}.bin'.format(output_filename))
        if os.path.exists(output_file):
            print('{} already exists'.format(output_file))
        else:
            print('Process {}'.format(output_file))
            input_spec = np.fromfile(wav_pair[0], np.float32).reshape(-1,1029)
            label_spec = np.fromfile(wav_pair[1], np.float32).reshape(-1,1029)
            features = np.hstack(align_specs(input_spec, label_spec))
            print('features', features.shape)
            with open(output_file, 'wb') as fp:
                fp.write(features.tostring())

#---------------------------data loader------------------------------------------
class Tanhize(object):
    ''' Normalizing `x` to [-1, 1] '''
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.xscale = xmax - xmin
    
    def forward_process(self, x):
        x = (x - self.xmin) / self.xscale
        if isinstance(x, np.ndarray):
            return np.clip(x, 0., 1.) * 2. - 1.
        else:
            try:
                return tf.clip_by_value(x, 0., 1.) * 2. - 1.
            except TypeError:
                raise TypeError('No such type {}'.format(type(x)))

    def backward_process(self, x):
        return (x * .5 + .5) * self.xscale + self.xmin


def read_pair(file_pattern, batch_size, record_lines=256, capacity=512, min_after_dequeue=128,
    num_threads=8, format='NCHW', normalizer=None):
    ''' 
    '''
    with tf.name_scope('InputSpectralFrame'):
        files = tf.gfile.Glob(file_pattern)
        filename_queue = tf.train.string_input_producer(files)

        record_bytes = FEAT_DIM * 2 * 4 * record_lines
        reader = tf.FixedLengthRecordReader(record_bytes)
        _, value = reader.read(filename_queue)
        value = tf.decode_raw(value, tf.float32)

        value = tf.reshape(value, [record_lines, 2*FEAT_DIM])
        feature = value[:,:SP_DIM]
        label = value[:,FEAT_DIM:SP_DIM+FEAT_DIM]
        
        feature = normalizer.forward_process(feature)
        label = normalizer.forward_process(label)
            
        if format == 'NCHW':
            feature = tf.reshape(feature, [-1, SP_DIM, 1])
            label = tf.reshape(label, [-1, SP_DIM, 1])
        elif format == 'NHWC':
            feature = tf.reshape(feature, [SP_DIM, -1, 1])
            label = tf.reshape(label, [SP_DIM, -1, 1])
        else:
            pass

        return tf.train.shuffle_batch(
            [feature, label],
            batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            num_threads=num_threads,
#             allow_smaller_final_batch=True,
            # enqueue_many=True,
            )

def read_pair_single_numpy(filename, record_lines=256, normalizer=None, fields='sp'):
    ''' 
    '''
    input_with_label = np.fromfile(filename, np.float32).reshape(-1, 2*FEAT_DIM)
    if input_with_label.shape[0] < record_lines:
        input_with_label_pad = input_with_label.copy()
    else:
        input_with_label_pad = np.zeros((record_lines,2*FEAT_DIM), np.float32)
        input_with_label_pad[:record_lines,:] = input_with_label[:record_lines,:]
    if fields == 'sp':
        feature, label = input_with_label_pad[:,:SP_DIM], input_with_label_pad[:,FEAT_DIM:SP_DIM+FEAT_DIM]
        if normalizer:
            feature = normalizer.forward_process(feature)
            label = normalizer.forward_process(label)
    elif fields == 'all':
        feature, label = input_with_label_pad[:,:FEAT_DIM], input_with_label_pad[:,FEAT_DIM:]
        if normalizer:
            feature = np.hstack((normalizer.forward_process(feature[:,:SP_DIM]), feature[:,SP_DIM:]))
            label = np.hstack((normalizer.forward_process(label[:,:SP_DIM]), label[:,SP_DIM:]))
    print feature.dtype
    return feature, label

def read_pair_batch_numpy(filenames, record_lines=256, normalizer=None, mode='train'):
    ''' 
    '''
    if mode == 'train':
        batch_inputs = [read_pair_single_numpy(filename, record_lines=record_lines, \
                                           normalizer=normalizer, fields='sp')[0] for filename in filenames]
        batch_labels = [read_pair_single_numpy(filename, record_lines=record_lines, \
                                           normalizer=normalizer, fields='sp')[1] for filename in filenames]
        return np.expand_dims(np.array(batch_inputs).astype(np.float32), axis=3),\
	   np.expand_dims(np.array(batch_labels).astype(np.float32), axis=3)
    elif mode == 'test':
        batch_inputs = [read_pair_single_numpy(filename, record_lines=record_lines, \
                                           normalizer=normalizer, fields='all')[0] for filename in filenames]
        batch_labels = [read_pair_single_numpy(filename, record_lines=record_lines, \
                                           normalizer=normalizer, fields='all')[1] for filename in filenames]
	batch_inputs = np.array(batch_inputs).astype(np.float32)
	batch_labels = np.array(batch_labels).astype(np.float32)
        return np.expand_dims(batch_inputs[:,:,:SP_DIM], axis=3),\
                np.expand_dims(batch_labels[:,:,:SP_DIM], axis=3),\
                batch_inputs[:,:,SP_DIM:], batch_labels[:,:,SP_DIM:]

#---------------------------alignments-------------------------------------------
def read_whole_features(file_pattern, num_epochs=1):
    '''
    Return
        `feature`: `dict` whose keys are `sp`, `ap`, `f0`, `en`, `speaker`
    '''
    files = tf.gfile.Glob(file_pattern)
    print('{} files found'.format(len(files)))
    filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    print("Processing {}".format(key))
    value = tf.decode_raw(value, tf.float32)
    value = tf.reshape(value, [-1, FEAT_DIM])
    return {
        'sp': value[:, :SP_DIM],
        'ap': value[:, SP_DIM : 2*SP_DIM],
        'f0': value[:, SP_DIM * 2],
        'en': value[:, SP_DIM * 2 + 1],
        'speaker': tf.cast(value[:, SP_DIM * 2 + 2], tf.int64),
        'filename': key,
	}

def align_specs(specfile1, specfile2, feat_dim=1026):
    distance, path = librosa.core.dtw(specfile1[:,:feat_dim].T, specfile2[:,:feat_dim].T)
    return specfile1[path[::-1,0],:], specfile2[path[::-1,1],:]



def baseline(wavfile, outfile, src, trg, fft_size=FFT_SIZE):
    ''' A baseline convertion from one person to another '''
    sample = extract(wavfile)
    sp_dim = fft_size // 2 + 1
    sample_converted = np.concatenate((sample[:,:sp_dim*2], convert_f0(sample[:,sp_dim*2],\
		    src, trg).reshape(-1,1), sample[:,sp_dim*2+1].reshape(-1,1)), axis=1)
    librosa.output.write_wav(outfile, pw2wav(sample_converted), sr=16000)

if __name__ == '__main__':
#    wavfile = 'data/VCTK-Corpus/wav48/p226/p226_006.wav'
#    outfile = 'test/p226_006_p225_006.wav'
#    baseline(wavfile, outfile, (4.8286185, 0.16229385), (5.2577271, 0.15516736))
# pc use only
#    extract_and_save_bin_to('data/matrix_sample/225', 'data/matrix_sample/', 225)
#    save_pair_to('data/matrix_sample/225/pair/train', './data/matrix_sample/225/etc/train_map.txt')
#    save_pair_to('data/matrix_sample/225/pair/test', './data/matrix_sample/225/etc/test_map.txt')
# matrix use
    dir_to_src = '/media/katefgroup/VCTK_data/VCTK-Corpus/'
    dir_to_bin = 'data/225/'
    extract_and_save_bin_to(dir_to_bin, dir_to_src, 225)
    save_pair_to('data/225/pair/train', './data/225/etc/train_map.txt')
    save_pair_to('data/225/pair/test', './data/225/etc/test_map.txt')

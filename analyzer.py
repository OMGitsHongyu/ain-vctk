import sys
import os
import numpy as np
import librosa
import pyworld as pw

FFT_SIZE = 1024
EPSILON = 1e-10
#FEAT_DIM = SP_DIM + SP_DIM + 1 + 1 + 1  # [sp, ap, f0, en, s]


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
    elif isinstance(filename, np.ndarray):
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
    mu_s, std_s = src
    mu_t, std_t = trg
    lf0 = np.where(f0 > 1., np.log(f0), f0)
    lf0 = np.where(lf0 > 1., (lf0 - mu_s)/std_s * std_t + mu_t, lf0)
    lf0 = np.where(lf0 > 1., np.exp(lf0), lf0)
    return lf0

def baseline(wavfile, outfile, src, trg, fft_size=FFT_SIZE):
    ''' A baseline convertion from one person to another '''
    sample = extract(wavfile)
    sp_dim = fft_size // 2 + 1
    sample_converted = np.concatenate((sample[:,:sp_dim*2], convert_f0(sample[:,sp_dim*2],\
		    src, trg).reshape(-1,1), sample[:,sp_dim*2+1].reshape(-1,1)), axis=1) 
    librosa.output.write_wav(outfile, pw2wav(sample_converted), sr=16000)

if __name__ == '__main__':
    wavfile = '../segan/data/VCTK-Corpus/wav48/p226/p226_006.wav'
    outfile = 'p226_006_p225_006.wav'
    baseline(wavfile, outfile, (4.8286185, 0.16229385), (5.2577271, 0.15516736))

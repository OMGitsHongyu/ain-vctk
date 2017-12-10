import sys
sys.path.remove('/home/hongyuz/.local/lib/python2.7/site-packages')
sys.path.append('/home/hongyuz/.local/lib/python2.7/site-packages')
import numpy as np
import pyworld as pw
import tensorflow as tf
from analyzer import pw2wav, read, read_whole_features

FILE_PATTERN = './data/matrix_sample/225/*a*/*bin'
SPEAKERS = range(225,251)
DATA_DIR = 'data/225/

def main():
    x = read_whole_features(DATA_DIR+'*a*/*bin')
    x_all = list()
    y_all = list()
    f0_all = list()
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        while True:
            try:
                features = sess.run(x)
                print('Processing {}'.format(features['filename']))
                x_all.append(features['sp'])
                y_all.append(features['speaker'])
                f0_all.append(features['f0'])
            finally:
                pass

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    f0_all = np.concatenate(f0_all, axis=0)

    for s in SPEAKERS:
        print('Speaker {}'.format(s))
        f0 = f0_all[np.int64(s) == y_all]
        print('  len: {}'.format(len(f0)))
        f0 = f0[f0 > 2.]
        f0 = np.log(f0)
        mu, std = f0.mean(), f0.std()
        print mu, std
        # Save as `float32`
        with open(DATA_DIR+'/etc/{}.npf'.format(s), 'wb') as fp:
            fp.write(np.asarray([mu, std]).tostring())


    # ==== Min/Max value ====
    # mu = x_all.mean(0)
    # std = x_all.std(0)
    q005 = np.percentile(x_all, 0.5, axis=0)
    q995 = np.percentile(x_all, 99.5, axis=0)

    # Save as `float32`
    with open(DATA_DIR+'/etc/xmin.npf', 'wb') as fp:
        fp.write(q005.tostring())

    with open(DATA_DIR+'/etc/xmax.npf', 'wb') as fp:
        fp.write(q995.tostring())

if __name__ == '__main__':
    main()

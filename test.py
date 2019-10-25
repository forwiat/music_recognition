from hparams import hyperparams as hp
import tensorflow as tf
import numpy as np
from model import Graph
from handle import get_FAR, get_FRR

def main(self):
    mode = 'test'
    g = Graph(mode=mode)
    print('{} graph loaded.'.format(mode))
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)) as sess:
        loaded = False
        try:
            try:
                print('try to load trained model in {} ...'.format(hp.model_dir))
                saver.restore(sess, tf.train.latest_checkpoint(hp.model_dir))
                loaded = True
            finally:
                if loaded is False:
                    print('load trained model failed in test, please check ...')
                    exit(0)
                total_y = np.zeros([hp.lab_size])
                total_y_hat = np.zeros([hp.lab_size])
                while 1:
                    y, y_hat = sess.run([g.y, g.y_hat])
                    total_y = np.concatenate((total_y, y), axis=0)
                    total_y_hat = np.concatenate((total_y_hat, y_hat), axis=0)
                total_y = total_y[1:, :]
                total_y_hat = total_y_hat[1:, :]
                EER = 0
                EER_thres = 0
                for i in range(1000):
                    threshold = i * 1.0 / 1000
                    FER = get_FRR(total_y, total_y_hat, threshold)
                    FAR = get_FAR(total_y, total_y_hat, threshold)
                    if abs(FER - FAR) < 1e-5:
                        EER_thres = threshold
                        EER = FER
                print('EER : {} EER_thres : {}'.format(EER, EER_thres))
        except:
            print('test over.')

if __name__ == '__main__':
    tf.app.run()

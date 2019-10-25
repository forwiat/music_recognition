import tensorflow as tf
from hparams import hyperparams as hp
from model import Graph
import os
import argparse
import numpy as np
from utils import get_spectrogram
import codecs

def pass_model(x, threshold):
    x = np.array(x)
    x = np.expand_dims(x, 0) # [1, frames, fft/2 + 1]
    mode = 'infer'
    g = Graph(mode=mode)
    print(f'{mode} graph loaded.')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hp.model_dir))
        y = sess.run([g.y_hat], feed_dict={g.x: x})
        y = np.array(y)
        y = np.squeeze(y)
    y[y < threshold] = 0
    y[y >= threshold] = 1
    return y

def get_id_label_dic():
    id_label_dic = {}
    lines = codecs.open(hp.label_info, 'r').readlines()[1:]
    for line in lines:
        label_id, label_name, _ = line.strip().split('\t')
        id_label_dic[int(label_id)] = label_name
    return id_label_dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path', '-i', type=str, help='The path of music passed through model, only supported wav.')
    parser.add_argument('--threshold', '-t', type=float, help='The threshold for class type.')
    parser.set_defaults(wav_path=None)
    parser.set_defaults(threshold=0.5)
    args = parser.parse_args()
    fpath = args.wav_path
    threshold = args.threshold
    if os.path.isfile(fpath) and os.path.basename(fpath)[-3:] == 'wav':
        id_label_dic = get_id_label_dic()
        input_x = get_spectrogram(fpath)
        y = pass_model(input_x, threshold)
        label_list = [id_label_dic[i] for i in range(len(y)) if y[i] == 1]
        print('该首歌标签为：')
        print(label_list)

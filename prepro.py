import numpy as np
import os
import tensorflow as tf
import codecs
import multiprocessing as mp
from hparams import hyperparams as hp
from tqdm import tqdm
from utils import get_spectrogram

def process(args):
    (wav_label_list, label_dic, cpu_id, mode) = args
    if mode == 'train':
        writer = tf.python_io.TFRecordWriter(os.path.join(hp.train_dir, '{}.tfrecord'.format(cpu_id)))
    elif mode == 'eval':
        writer = tf.python_io.TFRecordWriter(os.path.join(hp.eval_dir, '{}.tfrecord'.format(cpu_id)))
    else: # test
        writer = tf.python_io.TFRecordWriter(os.path.join(hp.test_dir, '{}.tfrecord'.format(cpu_id)))
    for line in tqdm(wav_label_list):
        _, wav_name, labels = line.strip().split('\t')
        labels = labels.split('|')
        wav_path = os.path.join(hp.wavs_dir, wav_name)
        train_x = get_spectrogram(wav_path)
        train_y = np.zeros(shape=[hp.lab_size])
        train_mask = np.zeros(shape=[hp.lab_size])
        for label in labels:
            train_y[int(label_dic[label][0])] = 1
        for label in labels:
            train_mask[int(label_dic[label][0])] = 1
        for label in labels:
            if len(label_dic[label]) <= 1:
                continue
            for reverse_label in label_dic[label][1:]:
                train_mask[int(label_dic[reverse_label][0])] = 1
        #--------write into tf record file----------#
        features = {}
        features['x'] = tf.train.Feature(float_list=tf.train.FloatList(value=train_x.reshape(-1)))
        features['x_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=train_x.shape))
        features['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=train_y.reshape(-1)))
        features['y_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=train_y.shape))
        features['mask'] = tf.train.Feature(float_list=tf.train.FloatList(value=train_mask.reshape(-1)))
        features['mask_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=train_mask.shape))
        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
        #--------write into tf record file----------#
    writer.close()

def get_label_dic():
    lines = codecs.open(hp.label_info, 'r').readlines()
    label_dic = {}
    for line in lines[1:]:
        print(line)
        label_id, label_name, reverse_labels = line.strip().split('\t')
        label_dic[label_name] = []
        label_dic[label_name].append(label_id)
        if reverse_labels == 'None':
            continue
        for reverse_label in reverse_labels.split('|'):
            label_dic[label_name].append(reverse_label)
    return label_dic

def get_wav_label_list(mode='train'):
    lines = codecs.open(hp.music_info, 'r').readlines()
    lines = lines[1:]
    if mode == 'train':
        start = 0
        end = hp.train_size
    elif mode == 'eval':
        start = hp.train_size
        end = hp.train_size + hp.eval_size
    else: # test
        start = hp.train_size + hp.eval_size
        end = hp.train_size + hp.eval_size + hp.test_size
    res = lines[start: end]
    return res

def multi_process(mode='train'):
    cpu_nums = mp.cpu_count()
    label_dic = get_label_dic()
    wav_label_list = get_wav_label_list(mode)
    pool = mp.Pool(cpu_nums)
    splits = [(wav_label_list[i::cpu_nums],
               label_dic,
               i,
               mode)
              for i in range(cpu_nums)]
    pool.map(process, splits)
    pool.close()
    pool.join()

def single_process(mode='train'):
    label_dic = get_label_dic()
    wav_label_list = get_wav_label_list(mode)
    splits = (wav_label_list, label_dic, 0, mode)
    process(splits)

def check_dirs():
    if not os.path.isdir(hp.train_dir):
        os.makedirs(hp.train_dir)
    if not os.path.isdir(hp.eval_dir):
        os.makedirs(hp.eval_dir)
    if not os.path.isdir(hp.test_dir):
        os.makedirs(hp.test_dir)

if __name__ == '__main__':
    check_dirs()
    if hp.prepro_mp:
        multi_process('train') # create train tf_data
        multi_process('eval') # create eval tf_data
        multi_process('test') # create test tf_data
    else:
        single_process('train') # create train tf_data
        single_process('eval') # create eval tf_data
        single_process('test') # create test tf_data

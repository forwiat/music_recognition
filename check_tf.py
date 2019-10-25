import numpy as np
import os
import tensorflow as tf
import codecs
import multiprocessing as mp
from hparams import hyperparams as hp
from tqdm import tqdm
from utils import get_spectrogram

def check():
    wav_label_list = get_wav_label_list(mode='test')
    label_dic = get_label_dic()
    cnt = 0
    for line in tqdm(wav_label_list):
        _, wav_name, labels = line.strip().split('\t')
        labels = labels.split('|')
        wav_path = os.path.join(hp.wavs_dir, wav_name)
        train_x = get_spectrogram(wav_path)
        train_y = np.zeros(hp.lab_size)
        train_mask = np.zeros(hp.lab_size)
        for label in labels:
            train_y[int(label_dic[label][0])] = 1
        for label in labels:
            train_mask[int(label_dic[label][0])] = 1
        for label in labels:
            if len(label_dic[label]) <= 1:
                continue
            for reverse_label in label_dic[label][1:]:
                train_mask[int(label_dic[reverse_label][0])] = 1
        print(train_y)
        print(train_mask)
        cnt += 1
        if cnt >= 1:
            break

def get_label_dic():
    lines = codecs.open(hp.label_info, 'r').readlines()
    label_dic = {}
    for line in lines[1:]:
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

if __name__ == '__main__':
    check()

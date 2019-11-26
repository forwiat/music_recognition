import numpy as np
import tensorflow as tf
from hparams import hyperparams as hp
import os
import librosa
def get_next_batch(mode='train'):
    def _parse(example_proto):
        dic = {
            'x': tf.VarLenFeature(dtype=tf.float32),
            'x_shape': tf.FixedLenFeature(shape=[2], dtype=tf.int64),
            'y': tf.FixedLenFeature(shape=[95], dtype=tf.float32),
            'y_shape': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            'mask': tf.FixedLenFeature(shape=[95], dtype=tf.float32),
            'mask_shape': tf.FixedLenFeature(shape=[1], dtype=tf.int64)
        }
        parsed_example = tf.parse_single_example(example_proto, dic)
        parsed_example['x'] = tf.sparse_tensor_to_dense(parsed_example['x'])
        parsed_example['x'] = tf.reshape(parsed_example['x'], parsed_example['x_shape'])
        parsed_example['y'] = tf.reshape(parsed_example['y'], parsed_example['y_shape'])
        parsed_example['mask'] = tf.reshape(parsed_example['mask'], parsed_example['mask_shape'])
        return parsed_example
    if mode == 'train':
        tf_dir = hp.train_dir
    elif mode == 'eval':
        tf_dir = hp.eval_dir
    elif mode == 'test':
        tf_dir = hp.test_dir
    else:
        raise Exception('no supported mode {} in get_next_batch function, please check ...'.format(mode))
    tf_files = [os.path.join(tf_dir, fname) for fname in os.listdir(tf_dir)]
    dataset = tf.data.TFRecordDataset(tf_files)
    parsed_dataset = dataset.map(_parse)
    shuffled_dataset = parsed_dataset.shuffle(buffer_size=hp.shuffle_size)
    batch_padded_dataset = shuffled_dataset.padded_batch(
        batch_size=hp.batch_size,
        padded_shapes={
            'x': [None, hp.f_size],
            'x_shape': [2],
            'y': [hp.lab_size],
            'y_shape': [1],
            'mask': [hp.lab_size],
            'mask_shape': [1]
        }
    )
    epoched_dataset = batch_padded_dataset.repeat(count=hp.num_epoches)
    iterator = epoched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element['x'], next_element['y'], next_element['mask']

def get_spectrogram(wav_path):
    y, _ = librosa.load(wav_path, sr=hp.sr)
    y, _ = librosa.effects.trim(y)
    fft_spectrogram = librosa.stft(y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
    mag_spectrogram = np.abs(fft_spectrogram)
    mag_spectrogram = 20 * np.log10(np.maximum(1e-5, mag_spectrogram))
    mag_spectrogram = np.clip((mag_spectrogram - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag_spectrogram = mag_spectrogram.T.astype(np.float32)
    n_frames, feat_size = mag_spectrogram.shape
    if n_frames >= hp.segment_length:
        start_index = np.random.randint(low=0, high=n_frames - hp.segment_length + 1)
        segmented_mag = mag_spectrogram[start_index: start_index + hp.segment_length, :]
    else:
        segmented_mag = np.concatenate((mag_spectrogram, np.zeros((hp.segment_length - n_frames, feat_size))), axis=0)
    return segmented_mag

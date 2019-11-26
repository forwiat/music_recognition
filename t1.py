# import numpy as np
# import tensorflow as tf
# import os
# from hparams import hyperparams as hp
# from model import Graph
# inputs = tf.placeholder(shape=[5, 4, 7], dtype=tf.float32)
# multi_cell = [tf.nn.rnn_cell.LSTMCell(size) for size in [10, 12, 15]]
# multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(multi_cell)
# a, b = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=inputs, dtype=tf.float32)
# b = b[-1][0]
# x = tf.placeholder(shape=[5, 7, 2], dtype=tf.float32)
# y = tf.placeholder(shape=[5, 7, 2], dtype=tf.float32)
# y_hat = tf.nn.softmax(x)
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     _x = np.random.normal(size=[5, 7, 2])
#     _y = np.random.randint(0, 1, size=[5, 7, 2])
#     _y_hat, _l = sess.run((y_hat, loss), feed_dict={x: _x, y: _y})
#     _l = np.array(_l)
#     _y_hat = np.array(_y_hat)
#     print(_l.shape)
#     print(_y_hat.shape)
#str = 'xieyongbin, nihao,'
#print(str.strip().strip(',').split(','))
#print(str.split(','))
# mode = 'train'
# g = Graph(mode=mode)
# print('{} graph loaded.'.format(mode))
# saver = tf.train.Saver()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)) as sess:
#     sess.run(tf.global_variables_initializer())
#     _x, _y, _mask, _, loss, summary, steps = sess.run([g.x, g.y, g.mask, g.train_op, g.loss, g.merged, g.global_step])
#     saver.save(sess, os.path.join(hp.model_dir, 'model_{}'.format(steps)))
#     _x = np.array(_x)
#     _y = np.array(_y)
#     _mask = np.array(_mask)
#     print(_x.shape)
#     print(_y.shape)
#     print(_mask.shape)
#     print('train mode \t steps  : {}, loss : {}'.format(steps, loss))
from pydub import AudioSegment
import librosa

def raw2wav(ori_path: str, aim_path: str, sr: int):
    y = AudioSegment.from_file(file=ori_path, format='pcm', sample_width=2, channels=1, frame_rate=sr)
    y.export(out_f=aim_path, format='wav')

def resample(ori_path: str, aim_path: str, ori_sr: int, aim_sr: int):
    y, _ = librosa.load(ori_path, sr=ori_sr)
    new_y = librosa.resample(y, orig_sr=ori_sr, target_sr=aim_sr)
    librosa.output.write_wav(aim_path, new_y, aim_sr)

ori_path = 'G:/check_wav/music-kdf.pcm'
aim_path = 'G:/check_wav/music-kdf.wav'
ori_sr = 16000
aim_sr = 16000
raw2wav(ori_path, aim_path, sr=ori_sr)
#resample(aim_path, aim_path, ori_sr, aim_sr)



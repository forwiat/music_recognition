from hparams import hyperparams as hp
import tensorflow as tf
from model import Graph
import os

def main(self):
    mode = 'train'
    g = Graph(mode=mode)
    print('{} graph loaded.'.format(mode))
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    with tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(hp.log_dir, sess.graph)
        loaded = False
        try:
            try:
                print('try to load trained model in {} ...'.format(hp.model_dir))
                saver.restore(sess, tf.train.latest_checkpoint(hp.model_dir))
                loaded = True
            finally:
                if loaded is False:
                    print('load trained model failed, start training with initializer ...')
                sess.run(tf.global_variables_initializer())
                while 1:
                    _, loss, summary, steps = sess.run([g.train_op, g.loss, g.merged, g.global_step])
                    print('train mode \t steps  : {}, loss : {}'.format(steps, loss))
                    writer.add_summary(summary, steps)
                    if steps % (hp.per_steps + 1) == 0:
                        saver.save(sess, os.path.join(hp.model_dir, 'model_{}'.format(steps)))
        except:
            print('train over.')

if __name__ == '__main__':
    tf.app.run()

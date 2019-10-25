import tensorflow as tf
from hparams import hyperparams as hp
from networks import lstm_3_layers
from utils import get_next_batch
import numpy as np

class Graph:
    def __init__(self, mode):
        self.mode = mode
        if self.mode in ['train', 'eval']:
            if self.mode == 'train' and len(hp.gpu_ids) > 1:
                self.multi_train()
            else:
                self.single_train()
            tf.summary.scalar('{}/loss'.format(self.mode), self.loss)
            self.merged = tf.summary.merge_all()
            self.t_vars = tf.trainable_variables()
            self.num_paras = 0
            for var in self.t_vars:
                var_shape = var.get_shape().as_list()
                self.num_paras += np.prod(var_shape)
            print("Total number of parameters : %r"%(self.num_paras))
        elif self.mode in ['test']:
            self.test()
        elif self.mode in ['infer']:
            self.infer()
        else:
            raise Exception('no supported mode in model __init__ function, please check ...')

    ###################################################################################
    #                                                                                 #
    #                                   multi gpu train                               #
    #                                                                                 #
    ###################################################################################

    def multi_train(self):
        def _assign_to_device(device, ps_device='/cpu:0'):
            PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
            def _assign(op):
                node_def = op if isinstance(op, tf.NodeDef) else op.node_def
                if node_def.op in PS_OPS:
                    return '/' + ps_device
                else:
                    return device
            return _assign

        def _average_gradients(tower_grads):
            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                grads = []
                for g, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                grad = tf.concat(grads, 0)
                grad = tf.reduce_mean(grad, 0)
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
            return average_grads

        with tf.device('/cpu:0'):
            self.x, self.y, self.mask = get_next_batch(self.mode)
            self.tower_grads = []
            self.global_step = tf.get_variable('global_step', initializer=0, dtype=tf.int32, trainable=False)
            self.lr = tf.train.exponential_decay(hp.lr, global_step=self.global_step,
                                                 decay_steps=hp.lr_decay_steps,
                                                 decay_rate=hp.lr_decay_rate)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            gpu_nums = len(hp.gpu_ids)
            per_batch = hp.batch_size // gpu_nums
            with tf.variable_scope('network'):
                for i in range(gpu_nums):
                    with tf.device(_assign_to_device('/gpu:{}'.format(hp.gpu_ids[i]), ps_device='/cpu:0')):
                        self._x = self.x[i * per_batch: (i + 1) * per_batch]
                        self._y = self.y[i * per_batch: (i + 1) * per_batch]
                        self._mask = self.mask[i * per_batch: (i + 1) * per_batch]
                        self.outputs = lstm_3_layers(self._x, num_units=hp.lab_size, bidirection=False,
                                                    scope='lstm_3_layers', reuse=tf.AUTO_REUSE)
                        # sigmoid, fifo-queue
                        self.y_hat = tf.nn.sigmoid(self.outputs)
                        tf.get_variable_scope().reuse_variables()
                        # loss
                        self.res = tf.square(self.y_hat - self._y)
                        self.loss = tf.reduce_mean(tf.multiply(self._mask, self.res))
                        grad = self.optimizer.compute_gradients(self.loss)
                        self.tower_grads.append(grad)
            self.tower_grads = _average_gradients(self.tower_grads)
            clipped = []
            for grad, var in self.tower_grads:
                grad = tf.clip_by_norm(grad, 5.)
                clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(clipped, global_step=self.global_step)

    ###################################################################################
    #                                                                                 #
    #                            single gpu train and eval                            #
    #                                                                                 #
    ###################################################################################

    def single_train(self):
        with tf.device('/gpu:{}'.format(hp.gpu_ids[0])):
            self.x, self.y, self.mask = get_next_batch(self.mode)
            self.global_step = tf.get_variable('global_step', initializer=0, dtype=tf.int32, trainable=False)
            self.lr = tf.train.exponential_decay(learning_rate=hp.lr, global_step=self.global_step,
                                                 decay_steps=hp.lr_decay_steps,
                                                 decay_rate=hp.lr_decay_rate)
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            with tf.variable_scope('network'):
                self.outputs = lstm_3_layers(self.x, num_units=hp.lab_size, bidirection=False,
                                            scope='lstm_3_layers', reuse=tf.AUTO_REUSE)
                self.y_hat = tf.nn.sigmoid(self.outputs)
            # loss
            self.res = tf.square(self.y_hat - self.y) # [B, classes]
            self.loss = tf.reduce_mean(tf.multiply(self.res, self.mask))
            self.grads = self.optimizer.compute_gradients(self.loss)
            clipped = []
            for grad, var in self.grads:
                grad = tf.clip_by_norm(grad, 5.)
                clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(clipped, global_step=self.global_step)

    ###################################################################################
    #                                                                                 #
    #                                  test data in cpu                               #
    #                                                                                 #
    ###################################################################################

    def test(self):
        with tf.device('/cpu:0'):
            self.x, self.y, self.mask = get_next_batch(mode=self.mode)
            with tf.variable_scope('network'):
                self.outputs = lstm_3_layers(self.x, num_units=hp.lab_size, bidirection=False,
                                           scope='lstm_3_layers', reuse=tf.AUTO_REUSE)
                self.y_hat = tf.nn.sigmoid(self.outputs)
            self.y_hat = tf.multiply(self.y_hat, self.mask)

    ###################################################################################
    #                                                                                 #
    #                             real data infer in cpu                              #
    #                                                                                 #
    ###################################################################################

    def infer(self):
        with tf.device('/cpu:0'):
            self.x = tf.placeholder(shape=[None, None, hp.f_size], dtype=tf.float32)
            with tf.variable_scope('network'):
                self.y_hat = lstm_3_layers(self.x, num_units=hp.lab_size, bidirection=False,
                                           scope='lstm_3_layers', reuse=tf.AUTO_REUSE)
                self.y_hat = tf.nn.sigmoid(self.y_hat)

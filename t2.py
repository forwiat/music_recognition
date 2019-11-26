import tensorflow as tf
a = tf.constant([[1, 1, 0], [0, 0, 1]], dtype=tf.float32)
b = tf.count_nonzero(a, axis=-1)
with tf.Session() as sess:
    _b = sess.run(b)
    print(_b)

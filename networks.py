import tensorflow as tf
def lstm_3_layers(inputs, num_units=None, bidirection=False, scope="lstm", reuse=tf.AUTO_REUSE):
    '''
    :param inputs: A 3-d tensor. [N, T, C]
    :param num_units: An integer. The last hidden units.
    :param bidirection: A boolean. If True, bidirectional results are concatenated.
    :param scope: A string. scope name.
    :param reuse: Boolean. whether to reuse the weights of a previous layer.
    :return: if bidirection is True, A 2-d tensor. [N, num_units * 2]
             else, A 2-d tensor. [N, num_units]
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if not num_units:
            num_units = inputs.get_shape().as_list[-1]
        # cellls = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 128, num_units]]
        cellls = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256, num_units]]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cellls)
        if bidirection:
            bw_cells = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256, num_units]]
            multi_bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cells)
            outputs, final_state = tf.nn.dynamic_rnn(multi_cell, multi_bw_cell, inputs=inputs, dtype=tf.float32)
            # outputs shape : top lstm outputs, ([N, T, num_units], [N, T, num_units])
            # lstm final_state : multi final state stack together, ([N, 2, num_units], [N, 2, num_units])
            return tf.concat(final_state, axis=2)[-1][0]
        outputs, final_state = tf.nn.dynamic_rnn(cell=multi_cell, inputs=inputs, dtype=tf.float32)
        # outputs shape : top lstm outputs, [N, T, num_units]
        # lstm final_state : multi final state stack together, [N, 2, num_units]
        return final_state[-1][0]


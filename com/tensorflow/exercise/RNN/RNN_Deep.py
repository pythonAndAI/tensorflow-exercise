import tensorflow as tf

#以下实现了一个深层循环神经网络的前向传播
if __name__ == "__main__":
    #定义一个基本的LSTM结构作为循环体的基础结构。深层循环神经网络也支持使用其他的循环体结构
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
    #通过MultiRNNCell类实现深层循环神经网络中每一个时刻的前向传播过程。其中number_of_laters表示有多少层，也就是结构中从Xt到Ht需要经过多少个LSTM结构。
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell("lstm_size") for _ in range("number_of_layers")])
    #和经典的循环神经网络一样，可以通过zero_state函数来获取初始状态
    state = stacked_lstm.zero_state("batch_size", tf.float32)

    for i in range(len("num_steps")):
        if i > 0: tf.get_variable_scope().reuse_variables()
        stacked_lstm_output, state = stacked_lstm("current_input", state)
        #下面是一个全连接和计算损失函数。和RNN_test中定义的一样
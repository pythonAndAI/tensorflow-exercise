import tensorflow as tf

#以下简单定义一个LSTM结构

if __name__ == "__main__":
    #定义一个LSTM结构，LSTM中使用的变量也会在该函数中自动被声明
    lstm = tf.nn.rnn_cell.BasicLSTMCell("lstm_hidden_size")

    #将LSTM中的状态初始化为全0数组。BasicLSTMCell类提供了zero_state函数来生成全零的初始状态。state是一个包含两个张量的LSTMStateTuple类，其中state.c和state.h分别对应了LSTM结构
    #中的c状态和h状态。和其他神经网络类似，在优化循环神经网络时，每次也会使用一个batch的训练样本。
    state = lstm.zero_state("batch_size", tf.float32)

    #定义损失函数
    loss = 0.0
    #虽然在测试时循环神经网络可以处理任意长度的序列，但是在训练中为了将循环网络展开成前馈神经网络。我们需要知道训练数据的序列长度。在以下代码中，用num_steps来表示这个长度，
    #后面将使用dynamic_rnn动态处理变长序列的方法
    for i in range("num_steps"):
        #在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量
        if i > 0 : tf.get_variable_scope().reuse_variables()

        #每一步处理时间序列中的一个时刻。将当前输入current_input(结构中的Xt)和前一时刻状态state(Ht-1和Ct-1)传入定义的LSTM结构可以得到当前LSTM的输出lstm_output(Ht)和更新后状态state
        #(Ht和Ct)。lstm_output用于输出给其他层，state用于输出给下一时刻，他们在dropout等方面可以有不同的处理方式。
        #处理每一个时刻的前向传播算结果。
        lstm_output, state = lstm("current_input", state)
        #将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出。
        final_output = fully_connected(lstm_output)
        #计算当前时刻输出的损失
        loss += calc_loss(final_output, "expected_output")
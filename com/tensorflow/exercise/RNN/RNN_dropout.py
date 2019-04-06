import tensorflow as tf

#实现RNN的dropout方法

if __name__ == "__main__":
    #定义LSTM结构
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
    #使用DropoutWrapper类来实现dropout功能。该类通过两个参数来控制dropout的概率。一个参数是input_keep_prob，它可以用来控制输入的dropout概率；另一个为output_keep_prob
    #它可以用来控制输出的dropout的概率。在使用了DropoutWrapper的基础上定义MultiRNNCell
    stat_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(lstm_cell("lstm_size")) for _ in range("number_of_layers")])
    #后面的定义和RNN_Deep定义一样
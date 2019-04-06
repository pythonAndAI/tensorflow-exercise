import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from com.tensorflow.exercise.logging import LOG

#利用循环神经网络实现对函数sinx取值的预测

#LSTM中隐藏层节点的个数
HIDDEN_SIZE = 30
#LSTM的层数
NUM_LAYERS = 2
#循环神经网络的训练序列长度
TIMESTEPS = 10
#训练轮数
TRAINING_STEPS = 10000
#batch大小
BATCH_SIZE = 32
#训练数据的个数
TRAINING_EXAMPLES = 10000
#测试数据的个数
TESTING_EXAMPLES = 1000
#采样间隔
SAMPLE_GAP = 0.01

def generate_data(sep):
    X = []
    y = []
    #序列的第i项和后面的TIMESTEPS-1项合在一起作为输入；第i + TIMESTEPS项作为输出。即用sin函数前面的TIMESTEPS个点的信息，预测第i + TIMESTEPS个点的函数值
    for i in range(len(sep) - TIMESTEPS):
        X.append([sep[i : i + TIMESTEPS]])
        y.append([sep[i + TIMESTEPS]])

    return np.array(X, np.float32), np.array(y, np.float32)

def lstm_model(X, y, is_training):
    #使用多层的LSTM结构
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    #使用Tensorflow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    #outputs是顶层LSTM在每一步的输出结果，它的维度是[batch_size, time, HIDDEN_SIZE]，在本问题中只关注最后时刻的输出结果
    output = outputs[:, -1, :]
    #对LSTM的输出在加上一层全连接层并计算损失，这里默认的损失为平均平方差损失函数
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    if not is_training:
        return predictions, None, None
    #计算损失函数
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    #创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer="Adagrad", learning_rate=0.1)
    return predictions, loss, train_op

def train(sess, train_x, train_y):
    #将训练数据以数据集的方式提供给计算图
    X, y = tf.data.Dataset.from_tensor_slices((train_x, train_y)).repeat().shuffle(1000).batch(BATCH_SIZE).make_one_shot_iterator().get_next()
    #调用模型，得到预测结果、损失函数，和训练操作
    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print("train step: " + str(i) + ", loss: " + str(l))

def run_eval(sess, test_x, test_y):
    X, y = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(1).make_one_shot_iterator().get_next()

    #调用模型得到预测结果。这里不需要输入真实的y值
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)

    #将预测结果存入一个数组
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)
    LOG.getlogger("labels1").info(labels)
    LOG.getlogger("predictions1").info(predictions)
    #计算rmse作为评估指标
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Mean Square Error is: %f" % rmse )

    LOG.getlogger("labels2").info(labels)
    LOG.getlogger("predictions2").info(predictions)

    #对预测的sin函数曲线进行绘图
    plt.figure()
    plt.plot(labels, label="real_sin")
    plt.plot(predictions, label="predictions")
    plt.legend()
    plt.show()

#用正玄函数生成训练和测试数据集合
#numpy.linspace函数可以创建一个等差序列的数组，它常用的参数有三个参数，第一个参数表示起始值，第二个参数表示终止值，第三个参数表示数列的长度。例如,linespace(1,10,10)
#产生的数组是array([1,2,3,4,5,6,7,8,9,10])
test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
LOG.getlogger("test start").info(test_start)
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
LOG.getlogger("test end").info(test_end)
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
LOG.getlogger("train X").info(train_X)
LOG.getlogger("train y").info(train_y)
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
LOG.getlogger("test X").info(test_X)
LOG.getlogger("test y").info(test_y)

with tf.Session() as sess:
    #训练模型
    train(sess, train_X, train_y)
    #使用训练好的模型对测试数据进行预测
    run_eval(sess, test_X, test_y)

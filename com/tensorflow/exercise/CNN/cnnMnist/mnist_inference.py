import tensorflow as tf
from com.tensorflow.exercise.CNN.cnnMnist import mnist_basis

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积神经网络的尺寸和深度
CONV1_SIZE = 5
CONV1_DEEP = 32

#第二层卷积神经网络的尺寸和深度
CONV2_SIZE = 5
CONV2_DEEP = 64

#全连接层的节点个数
FC_SIZE = 512

#输入为batch*28*28*1
def inference(input_tense, train, regularizer):
    with tf.variable_scope("layer1-conv1"):
        #第一层卷积层的权重，过滤器的大小为5*5，当前深度为1，输出深度为32，weigths大小为5*5*1*32
        conv1_weigths = mnist_basis.get_weigths("weigth", CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP)
        # 第一层卷积层的偏置，大小为1*32
        conv1_biases = mnist_basis.get_biases("biases", CONV1_DEEP)
        # 第一层卷积层的输出，因为使用了全0填充，且过滤器步长为1，所以卷积层的大小为batch*28*28*32
        conv1_relu1 = mnist_basis.get_conv2d(input_tense, conv1_weigths, conv1_biases)

    with tf.variable_scope("layer2-pool1"):
        #第一层的池化层，过滤器尺寸为2*2，步长为2，且使用全0填充，输入为batch*28*28*32，所以池化层的大小为batch*14*14*32
        pool1 = mnist_basis.get_pool(conv1_relu1, 2, 2)

    with tf.variable_scope("layer3-conv2"):
        #第二层的卷积层的权重，过滤器的尺寸为5*5，当前层深度为32，卷积层深度为64，大小为5*5*32*64
        conv2_weigths = mnist_basis.get_weigths("weigth", CONV2_SIZE, CONV1_DEEP, CONV2_DEEP)
        #第二层的卷积层的偏置，大小为1*64
        conv2_biases = mnist_basis.get_biases("biases", CONV2_DEEP)
        #第二层卷积层的输出，因为使用了全0填充，且步长为1，所以卷积层的大小为batch*14*14*64
        #先计算conv1，加上biases，再利用激活函数
        conv2_relu2 = mnist_basis.get_conv2d(pool1, conv2_weigths, conv2_biases)

    with tf.variable_scope("layer4-pool2"):
        #第二个最大池化层，过滤器尺寸为2, 步长为2，且使用全0填充，输入为batch*14*14*64，所以池化层的大小为batch*7*7*64
        pool2 = mnist_basis.get_pool(conv2_relu2, 2, 2)

    #将第四层池化层的输出转化为第五层全连接的输入格式，第四层的输出为batch*7*7*64的矩阵，然而第五层全连接层需要的输入格式为向量，所以在这里需要将这个7*7*64的矩阵拉直成为
    #一个向量。pool.get_shape()函数可以得到第四层输出矩阵的维度而不需要手工计算，注意因为每一层神经网络的输入输出都为一个batch的矩阵，所以这里得到的维度也包含了一个batch中数据的个数
    #得到池化层的维度[batch, 7, 7, 64]
    pool_shape = pool2.get_shape().as_list()
    #nodes=7*7*64=3136
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #把[batch, 7, 7, 64]的四维矩阵转化为[batch， 3136]的二维矩阵
    reshape = tf.reshape(pool2, [pool_shape[0], nodes])

    #定义第五层的前向传播算法.这一层的输入为[batch， 3136]的二维矩阵，输出为[batch,512]的二维矩阵。引入dropout概念，dropout在训练时会随机将部分节点的输出改为0，
    #dropout可以避免过拟合问题，从而使得模型在测试数据上的效果更好。dropout一般只在全连接层而不是卷积层或者池化层使用
    #输出为[batch, 512]的二维矩阵
    with tf.variable_scope("layer5-fc1"):
        fcl_weigths = tf.get_variable(name="weigths", shape=[nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))

        #只用全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fcl_weigths))

        fcl_biases = tf.get_variable(name="biases", shape=[FC_SIZE], initializer=tf.constant_initializer(0.1))
        fcl = tf.nn.relu(tf.matmul(reshape, fcl_weigths) + fcl_biases)
        if train:
            fcl = tf.nn.dropout(fcl, 0.5)

    #定义第六层全连接层并实现前向传播算法，输入为[batch, 512]的二维矩阵，输出层为10，这一层的输出通过Softmax之后就得到了最后的分类结果
    #输出为[batch, 10]的二维矩阵
    with tf.variable_scope("layer6-fc2"):
        fc2_weigths = tf.get_variable(name="weigth", shape=[FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weigths))

        fc2_biases = tf.get_variable(name="biases", shape=[NUM_LABELS], initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fcl, fc2_weigths) + fc2_biases
        #返回第六层的输出
    return logit










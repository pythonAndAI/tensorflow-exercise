import tensorflow as tf

'''
激活函数实现去线性化
case中定义的神经元结构的输出为所有输入的加权和，这导致整个神经网络是一个线性模型。如果将每一个神经元(也就是神经网络中的每一个节点)
的输出通过一个非线性函数，那么整个神经网络的模型也就不再是线性的了。这个非线性函数就是激活函数。可以使用激活函数实现神经网络的去线性化
目前Tensorflow提供7种不同的非线性激活函数，tf.nn.relu、tf.sigmoid、tf.tanh是其中比较常用的几种
更多请参考http://www.tensorfly.cn/tfdoc/api_docs/python/nn.html
'''

def activa_func():
    w1 = tf.Variable(tf.random_normal([2, 3]))
    w2 = tf.Variable(tf.random_normal([3, 1]))

    x = tf.placeholder(dtype=tf.float32, shape=(1, 2))
    biases = tf.constant(1, dtype=tf.float32)

    a = tf.nn.relu(tf.matmul(x, w1) + biases)
    y = tf.nn.relu(tf.matmul(a, w2) + biases)


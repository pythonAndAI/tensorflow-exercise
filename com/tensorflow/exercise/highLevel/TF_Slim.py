import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#使用Tensorflow-Slim在Mnist数据集上实现LeNet5模型
#从以下代码可以看出，Tensorflow-Slim主要的作用是使模型定义更加简洁，基本上每一层网络可以通过一句话来实现。除了对单层网络结构，
#Tensorflow-Slim还对数据预处理、损失函数、学习过程、测试过程等都提供了高层封装。Tensorflow-Slim最特别的一个地方是它对一些标准的神经网络模型进行了封装，比如VGG、Inception以及ResNet。
#而且Google开源的训练好的图像分类模型基本都是通过Tensorflow-Slim实现的。

def lenet5(inputs):
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])
    #定义第一个卷积层，当前卷积层深度为32，过滤器尺寸为5，步长为1，默认使用全0填充
    net = slim.conv2d(inputs, 32, [5, 5], scope="layer1-conv")
    # 定义第一个最大池化层，过滤器尺寸为2，步长为2
    net = slim.max_pool2d(net, 2, stride=2, scope="layer2-max-pool")
    # 定义第二个卷积层，当前卷积层深度为64，过滤器尺寸为5，步长为1，默认使用全0填充
    net = slim.conv2d(net, 64, [5, 5], scope="layer3-conv")
    # 定义第二个最大池化层，过滤器尺寸为2，步长为2
    net = slim.max_pool2d(net, 2, stride=2, scope="layer4-max-pool")
    #直接使用Tensorflow-Slim封装好的flatten函数将4维矩阵转为2维，这样可以方便后面的全连接层的计算。通过封装好的函数，用户不需要自己计算通过卷积层之后矩阵的大小
    net = slim.flatten(net, scope="flatten")
    net = slim.fully_connected(net, 500, scope="layer5")
    net = slim.fully_connected(net, 10, scope="output")
    return net

def train(mnist):
    #定义输入
    X = tf.placeholder(tf.float32, [None, 784], name="X-input")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")
    y = lenet5(X)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(10000):
            xs, ys = mnist.train.next_batch(100)
            _, loss_value = sess.run([train_op, loss], feed_dict={X: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training steps, loss on training batch is %g" % (i, loss_value))

def main(argv=None):
    mnist = input_data.read_data_sets("E:\\Alls\\软件\\tensorflow-mnist", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()

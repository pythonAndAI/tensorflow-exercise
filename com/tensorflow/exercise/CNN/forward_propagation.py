import tensorflow as tf

'''
以下为一个CNN前向传播的例子
'''

if __name__ == "__main__":
    #生成过滤器的权重变量。卷积层的参数个数只和过滤器的尺寸，深度以及当前层节点矩阵的深度有关。所以这里声明的参数变量是一个四维矩阵，
    #前面两个维度代表了过滤器的尺寸，第三个维度表示当前层的深度，第四个维度表示过滤器的深度
    weigths = tf.get_variable(name="w1", shape=[5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))

    #和卷积层的权重类似，当前层矩阵上不同位置的偏置项也是共享的，所以总共有下一个深度个不同的偏置项。
    biases = tf.get_variable(name="b", shape=[16], initializer=tf.constant_initializer(0.1))

    #tf.nn.conv2d提供了一个非常方便的函数来实现卷积层前向传播的算法。这个函数的第一个输入为当前层的节点矩阵，注意这个矩阵是一个四维矩阵，
    # 后面三个维度对应一个节点矩阵，第一维对应一个输入batch。比如输入层input[0,:,:,:]表示第一张图片，input[1,:,:,:]表示第二张图片，以此类推。
    #tf.nn.conv2d的第二个参数提供了卷积层的权重，tf.nn.conv2d的第三个参数为不同维度上的步长。虽然第三个参数提供的是一个长度为4的数组，但是第一维和最后一维的数字要求一定是1.
    #这是因为卷积层的步长只对矩阵的长和宽有效。tf.nn.conv2d的最后一个参数是填充(padding)的方法，Tensorflow中提供SAME或是VALID两种选择。其中SAME表示添加全0填充，VALID表示不添加
    # conv = tf.nn.conv2d("[1.0 ,2.0, 3.0, 4.0]", weigths, strides=[1, 1, 1, 1], padding="SAME")

    #tf.nn.bias_add提供了一个方便的函数给每一个节点加上偏置项。注意这里不能直接使用加法，因为矩阵上不同位置上的节点都需要加上同样的偏置项。
    #比如，虽然下一层神经网络的大小为2x2，但是偏置项只有一个数(因为深度为1)，而2x2矩阵中的每一个值都需要加上这个偏置项。
    # bias = tf.nn.bias_add(conv, biases)

    #将计算结果通过RELU激活函数完成去线性化
    # active_conv = tf.nn.relu(bias)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(weigths))




import tensorflow as tf
from com.tensorflow.exercise.logging import LOG

def test_slim():
    slim = tf.contrib.slim
    input = tf.Variable(tf.truncated_normal([1, 28, 28, 1]))
    #使用Tensorflow-Slim实现卷积层。通过Tensorflow-Slim可以在一行中实现一个卷积层的前向传播算法。slim.conv2d函数有三个参数是必填的，第一个参数为输入矩阵的大小，
    #第二个参数是当前卷积层过滤器的深度，第三个参数是过滤器的尺寸。可选的参数有过滤器移动的步长、是否使用全0填充、激活函数的选择以及变量的命名空间等
    net = slim.conv2d(input, 32, [3, 3])
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(net))
        print(sess.run(net[0][0]))
        print(net.shape)

#此处练习Inception-v3.png的最后一个Inception结构
#假设输入是个[batch, 28, 28, 1]的四维矩阵。
def inception_test(input):
    #加载slim库
    slim = tf.contrib.slim
    #slim.arg_scope函数可以用于设置默认的参数取值。slim.arg_scope函数的第一个参数是一个函数列表，在这个列表中的函数将使用默认的参数取值。比如通过如下的定义，调用
    #slim.conv2d(input, 32, [1, 1])函数时会自动加上stride=1和padding="SAME"的参数。如果在函数调用时指定了stride，那么这里设置的默认值就不会使用。通过这种方式可以进一步减少冗余的代码
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding="SAME"):
        #为每一个inception模型声明一个统一的变量命名空间
        with tf.variable_scope("Mixed_7c"):
            #给inception模型中每一个路径声明一个命名空间
            with tf.variable_scope("Branch_0"):
                #实现一个过滤器边长为1，深度为320的卷积层。输出为[batch, 28, 28, 320]
                branch_0 = slim.conv2d(input, 320, [1, 1], scope="Conv2d_0a_1x1")
                LOG.getlogger("branch_0").info(branch_0.shape)

            #Inception模型的第二条路径，这条计算路径上的结构本省也是一个Inception结构
            with tf.variable_scope("Branch_1"):
                #实现一个过滤器边长为1，深度为384的卷积层.输出为[batch, 28, 28, 384]
                branch_1 = slim.conv2d(input, 384, [1, 1], scope="Conv2d_0a_1x1")
                LOG.getlogger("branch_1_1").info(branch_1.shape)
                #concat函数可以将多个矩阵拼接起来。第一个参数指定了拼接的维度，这里的3代表了矩阵是在深度这个维度上进行的拼接。输出为[batch, 28, 28, 768]
                branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope="Conv2d_0b_1x3"), slim.conv2d(branch_1, 384, [3, 1], scope="Conv2d_0c_3x1")], 3)
                LOG.getlogger("branch_1_2").info(branch_1.shape)

            #Inception模型的第三条路径，这条计算路径上的结构本省也是一个Inception结构
            with tf.variable_scope("Branch_2"):
                #实现一个过滤器边长为1，深度为448的卷积层.输出为[batch, 28, 28, 448]
                branch_2 = slim.conv2d(input, 448, [1, 1], scope="Conv2d_0a_1x1")
                LOG.getlogger("branch_2_1").info(branch_2.shape)
                #输出为[batch, 28, 28, 384]
                branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope="Conv2d_0b_3x3")
                LOG.getlogger("branch_2_2").info(branch_2.shape)
                #concat函数可以将多个矩阵拼接起来。第一个参数指定了拼接的维度，这里的3代表了矩阵是在深度这个维度上进行的拼接。输出为[batch, 28, 28, 768]
                branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope="Conv2d_0c_1x3"), slim.conv2d(branch_2, 384, [3, 1], scope="Conv2d_0d_3x1")], 3)
                LOG.getlogger("branch_2_3").info(branch_2.shape)

            #Inception模型的第四条路径，这条计算路径上的结构本省也是一个Inception结构
            with tf.variable_scope("Branch_3"):
                #实现一个过滤器边长为1，深度为320的卷积层.输出为[batch, 28, 28, 1]
                branch_3 = slim.avg_pool2d(input, [3, 3], scope="AvgPool_0a_3x3")
                LOG.getlogger("branch_3_1").info(branch_3.shape)
                #输出为[batch, 28, 28, 192]
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope="Conv2d_0b_1x1")
                LOG.getlogger("branch_3_2").info(branch_3.shape)

            #当前Inception模型的最后输出是由上面4个计算结果拼接得到的,输出为[batch, 28, 28, 2048]
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            LOG.getlogger("final").info(net.shape)

    return net


if __name__ == "__main__":
    # test_slim()
    input = tf.Variable(tf.truncated_normal([1, 28, 28, 1]))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        inception_test(input)


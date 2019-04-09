import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from com.utils import Log_Util
import matplotlib.pyplot as plt

def getLogger(name):
    return Log_Util.getlogger(name)

def get_weigths(name, filter_size, current_depth, output_depth, regularizer=None):
    wegiths = tf.get_variable(name=name, shape=[filter_size, filter_size, current_depth, output_depth], initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 只用全连接层会用到正则化
    # if regularizer !=None:
    #     tf.add_to_collection(name, regularizer(wegiths))
    return wegiths

def get_biases(name, shape, num=0.0):
    biasis = tf.get_variable(name=name, shape=[shape], initializer=tf.constant_initializer(num))
    return biasis

def get_conv2d(input, weigths, biases, filter_step=1, padding="SAME"):
    conv = tf.nn.conv2d(input, weigths, strides=[1, filter_step, filter_step, 1], padding=padding)
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases))
    return relu

def get_pool(conv, filter_size, filter_step, padding="SAME"):
    pool = tf.nn.max_pool(conv, ksize=[1, filter_size, filter_size, 1], strides=[1, filter_step, filter_step, 1], padding=padding)
    return pool

def drawing(num, isOK=True):
    if isOK:
        plt.imshow(num)
        plt.show()

if __name__ == "__main__":
    mnist = input_data.read_data_sets("E:\\Alls\\软件\\tensorflow-mnist", one_hot=True)
    #大小为28个28*1(1*28*28*1)
    xs, ys = mnist.train.next_batch(100)
    # input = np.reshape(mnist.train.images[0], (1, 28, 28, 1))。真实数字时用np.reshape，tensorflow时用tf.reshape
    input = np.reshape(xs, (100, 28, 28, 1))
    # drawing(np.reshape(mnist.train.images[0], (28, 28)))

    #大小为5*5*1*32--(有5个1*32，再有5个的5的1*32)
    weigth_conv1 = get_weigths("layer-conv1", 5, 1, 32)
    #大小为1*32
    biases_conv = get_biases("layers-conv1", 32)
    #大小为1*28*28*32
    conv2 = get_conv2d(input, weigth_conv1, biases_conv)
    #大小为1*14*14*32
    pool = get_pool(conv2, 2, 2)

    pool_shape = pool.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshape = tf.reshape(pool, [pool_shape[0], nodes])
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # result_conv = sess.run(conv2[0])
        # for i in range(27):
        #     drawing(result_conv[i])
        # Log_Util.getlogger("weigth shape").info(weigth_conv1.shape)
        # Log_Util.getlogger("weigth one").info(len(sess.run(weigth_conv1)))
        # Log_Util.getlogger("weigth two").info(len(sess.run(weigth_conv1[0])))
        # Log_Util.getlogger("weigth three").info(len(sess.run(weigth_conv1[0][0])))
        # Log_Util.getlogger("weigth five").info(len(sess.run(weigth_conv1[0][0][0])))
        #
        # Log_Util.getlogger("conv2 shape").info(conv2.shape)
        # Log_Util.getlogger("conv2 one").info(len(sess.run(conv2)))
        # Log_Util.getlogger("conv2 two").info(len(sess.run(conv2[0])))
        # Log_Util.getlogger("conv2 three").info(len(sess.run(conv2[0][0])))
        # Log_Util.getlogger("conv2 five").info(len(sess.run(conv2[0][0][0])))
        #
        # Log_Util.getlogger("pool shape").info(pool.shape)
        # Log_Util.getlogger("pool one").info(len(sess.run(pool)))
        # Log_Util.getlogger("pool two").info(len(sess.run(pool[0])))
        # Log_Util.getlogger("pool three").info(len(sess.run(pool[0][0])))
        # Log_Util.getlogger("pool five").info(len(sess.run(pool[0][0][0])))
        Log_Util.getlogger("pool shape2").info(pool_shape)
        Log_Util.getlogger("pool nodes").info(nodes)
        Log_Util.getlogger("pool reshape").info(reshape)
        # logger.info(len(sess.run(conv2[0])))
        # logger.info(sess.run(pool))
        # logger.info(sess.run(pool[0][0]))
        # logger.info(len(sess.run(pool[0])))

    pass
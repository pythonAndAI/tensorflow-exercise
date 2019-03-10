import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from com.tensorflow.exercise.logging import LOG
import matplotlib.pyplot as plt

def getLogger(name):
    return LOG.getlogger(name)

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

def drawing(num):
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
        # LOG.getlogger("weigth shape").info(weigth_conv1.shape)
        # LOG.getlogger("weigth one").info(len(sess.run(weigth_conv1)))
        # LOG.getlogger("weigth two").info(len(sess.run(weigth_conv1[0])))
        # LOG.getlogger("weigth three").info(len(sess.run(weigth_conv1[0][0])))
        # LOG.getlogger("weigth five").info(len(sess.run(weigth_conv1[0][0][0])))
        #
        # LOG.getlogger("conv2 shape").info(conv2.shape)
        # LOG.getlogger("conv2 one").info(len(sess.run(conv2)))
        # LOG.getlogger("conv2 two").info(len(sess.run(conv2[0])))
        # LOG.getlogger("conv2 three").info(len(sess.run(conv2[0][0])))
        # LOG.getlogger("conv2 five").info(len(sess.run(conv2[0][0][0])))
        #
        # LOG.getlogger("pool shape").info(pool.shape)
        # LOG.getlogger("pool one").info(len(sess.run(pool)))
        # LOG.getlogger("pool two").info(len(sess.run(pool[0])))
        # LOG.getlogger("pool three").info(len(sess.run(pool[0][0])))
        # LOG.getlogger("pool five").info(len(sess.run(pool[0][0][0])))
        LOG.getlogger("pool shape2").info(pool_shape)
        LOG.getlogger("pool nodes").info(nodes)
        LOG.getlogger("pool reshape").info(reshape)
        # logger.info(len(sess.run(conv2[0])))
        # logger.info(sess.run(pool))
        # logger.info(sess.run(pool[0][0]))
        # logger.info(len(sess.run(pool[0])))

    pass
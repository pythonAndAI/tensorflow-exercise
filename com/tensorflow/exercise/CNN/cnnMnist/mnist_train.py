import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from com.tensorflow.exercise.CNN.cnnMnist import mnist_inference
import numpy as np
import os
from com.tensorflow.exercise.logging import LOG

#配置神经网络的参数
BATCH_SIZE = 100
REGULARIZER_RATE = 0.0001
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
TRAINING_STEPS = 30000
#模型保存的路径和文件名
MODEL_SAVE_PATH = "E:\Alls\软件\model\save"
MODEL_NAME = "model.ckpt"

def train(mnist):
    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name="x-input")
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.NUM_LABELS], name="y-input")

    #定义正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    #获取前向传播结果
    y = mnist_inference.inference(x, False, regularizer)

    #定义滑动平均模型、损失函数、指数衰减法、优化器
    global_step = tf.Variable(0, trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    average_op = variable_average.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #持久化模型
    saver = tf.train.Saver()
    with tf.control_dependencies([train_step, average_op]):
        train_op = tf.no_op("train")

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #训练过程
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            input_tensor = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            print(i)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: input_tensor, y_: ys})
            if i % 1000 == 0:
                # print("After %d training step(s),loss on  training " "batch is %g ." % (step, loss_value))
                LOG.getlogger("loss").info(loss_value)
                LOG.getlogger("step").info(step)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("E:\\Alls\\软件\\tensorflow-mnist", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()






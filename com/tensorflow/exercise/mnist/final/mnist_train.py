import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
#加载mnist_inference.py中定义的常量和前向传播的函数
from com.tensorflow.exercise.mnist.final import mnist_inference

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
    x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.INPUT_NODES], name="x-input")
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.OUTPUT_NODES], name="y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    #直接使用mnist_inference中定义的前向传播过程。
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    #定义损失函数、滑动平均模型、指数学习率。
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_steps, variable_average_op]):
        train_op = tf.no_op(name="train")

    #初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #在训练过程中不在测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step, learn = sess.run([train_op, loss, global_step, learning_rate], feed_dict={x: xs, y_: ys})
            #每1000轮保存一次模型
            if i % 1000 == 0:
                #输出当前的训练情况，这里只输出了模型在当前训练batch上的损失函数大小，通过损失函数的大小可以大概了解训练的情况，在验证数据集上的正确率信息会有一个单独的程序来生成
                print("After %d training step(s),loss on  training " "batch is %g , learning is %g." % (step, loss_value, learn))
                #保存当前的模型，注意这里给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，比如"model.ckpt-1000"表示训练1000轮之后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("E:\\Alls\\软件\\tensorflow-mnist", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()
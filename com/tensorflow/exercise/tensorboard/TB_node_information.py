import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from com.tensorflow.exercise.mnist.final import mnist_inference
from com.utils import Cmd_Util
from com.utils import File_Util

BATCH_SIZE = 100
REGULARIZER_RATE = 0.0001
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
TRAINING_STEPS = 30000

#基于TB_mnist.py的结构，展示TensorFlow计算图上每个节点的基本信息以及运行时消耗的时间和空间。
def train(mnist, path):
    #将处理输入数据的计算都放在名字为“input”的命名空间下。
    with tf.name_scope("input"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.INPUT_NODES], name="x-input")
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.OUTPUT_NODES], name="y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    #将处理滑动平均相关的计算都放在名为moving_average的命名空间下
    with tf.name_scope("moving_average"):
        variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_average_op = variable_average.apply(tf.trainable_variables())

    # 将处理损失函数相关的计算都放在名为loss_function的命名空间下
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 将定义学习率、优化方法以及每一轮训练需要执行的操作都放在名为train_step的命名空间下
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
        train_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_steps, variable_average_op]):
            train_op = tf.no_op(name="train")

    writer = tf.summary.FileWriter(path, tf.get_default_graph())

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            if i % 1000 == 0:
                #配置运行时需要记录的信息
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #运行时记录运行信息的proto
                run_metadata = tf.RunMetadata()
                _, loss_value, step, learn = sess.run([train_op, loss, global_step, learning_rate],
                                                      feed_dict={x: xs, y_: ys}, options=run_options, run_metadata=run_metadata)
                #'step%03d' % i 会在session runs中显示
                writer.add_run_metadata(run_metadata, 'step%03d' % i)
                print("After %d training step(s),loss on  training " "batch is %g , learning is %g." % (step, loss_value, learn))
            else:
                _, loss_value, step, learn = sess.run([train_op, loss, global_step, learning_rate],
                                                      feed_dict={x: xs, y_: ys})

    writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets("E:\\Alls\\软件\\tensorflow-mnist", one_hot=True)
    path = File_Util.get_path("E:\\Alls\\software\\tensorboard")
    File_Util.remove_file(path)
    train(mnist, path)
    Cmd_Util.run_tensorboard()

if __name__ == "__main__":
    tf.app.run()

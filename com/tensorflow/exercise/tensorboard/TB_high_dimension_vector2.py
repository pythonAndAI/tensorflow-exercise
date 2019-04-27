import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from com.utils import constant_Util
from com.utils import Cmd_Util

#加载用于生成PROJECTOR日志的帮助函数
from tensorflow.contrib.tensorboard.plugins import projector
from com.tensorflow.exercise.mnist.final import mnist_inference

BATCH_SIZE = 100
REGULARIZER_RATE = 0.0001
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
TRAINING_STEPS = 10000

LOG_DIR = os.getcwd()
TENSOR_NAME = "FINAL_LOGITS"

def train(mnist):
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

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step, learn = sess.run([train_op, loss, global_step, learning_rate], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s),loss on  training " "batch is %g , learning is %g." % (step, loss_value, learn))

        #计算MNIST测试数据对应的输出层矩阵
        final_result = sess.run(y, feed_dict={x: mnist.test.images})
    #返回输出层矩阵的值
    return final_result

#生成可视化最终输出层向量所需要的日志文件
def visualisation(final_result):
    #使用一个新的变量来保存最终输出层向量的结果。因为embedding是通过TensorFlow中变量完成的，所以PROJECTOR可视化的都是TensorFlow中的变量。于是这里需要新定义一个变量来保存输出层向量的取值。
    y = tf.Variable(final_result, name=TENSOR_NAME)
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    #通过projector.ProjectorConfig类来帮助生成日志文件
    config = projector.ProjectorConfig()
    #增加一个需要可视化的embeddings.add结果
    embedding = config.embeddings.add()
    #指定这个embedding结果对应的TensorFlow变量名称
    embedding.tensor_name = y.name

    #指定embedding结果所对应的原始数据信息。比如这里指定的就是每一张MNIST测试图片对应的真实类型。在单词向量中可以是单词ID对应的单词。这个文件是可选的。如果没有指定那么向量就没有标签
    embedding.metadata_path = constant_Util.META_FILE

    #指定sprite图像。这个也是可选的，如果没有提供sprite图像，那么可视化的结果每一个点就是一个小圆点，    而不是具体的图片
    embedding.sprite.image_path = constant_Util.SPRITE_FILE

    #在提供sprite图像时，通过single_image_dim可以指定单张图片的大小，这将用于从sprite图像中截取正确的原始图片
    embedding.sprite.single_image_dim.extend([28, 28])

    #将projector所需要的内容写入日志文件
    projector.visualize_embeddings(summary_writer, config)

    #生成会话，初始化新声明的变量并将需要的日志信息写入文件
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model"), TRAINING_STEPS)

    summary_writer.close()

#主函数先调用模型训练的过程，在使用训练好的模型来处理MNIST测试数据，最后将得到的输出层矩阵输出到PROJECTOR需要的日志文件中。
def main(argv=None):
    mnist = input_data.read_data_sets(constant_Util.MNIST_PATH, one_hot=True)
    final_result = train(mnist)
    visualisation(final_result)
    Cmd_Util.run_tensorboard(path=LOG_DIR)

if __name__ == "__main__":
    tf.app.run()
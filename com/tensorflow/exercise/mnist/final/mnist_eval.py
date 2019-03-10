import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
#加载mnist_inference和mnist_train中定义的常量和函数
from com.tensorflow.exercise.mnist.final import mnist_inference
from com.tensorflow.exercise.mnist.final import mnist_train

#准确率达到98.4%
#每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SIZE = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.INPUT_NODES], name="x-input")
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.OUTPUT_NODES], name="y-input")

        validation_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 直接通过调用封装好的函数来计算前向传播的结果。因为测试时不关注正则化损失的值，所以这里用于计算正则化损失的函数设置为None
        y = mnist_inference.inference(x, None)

        # 计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平均值了。这样就可以完全共用mnist_inference中定义的前向传播过程
        variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        # 每隔EVAL_INTERVAL_SIZE秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                print("ckpt is", ckpt)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    print("global_step", global_step)
                    accuracy_score = sess.run(accuracy, feed_dict=validation_feed)
                    print(
                        "After %s training step(s), validation accuracy = %g " % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SIZE)

def main(argv=None):
    mnist = input_data.read_data_sets("E:\\Alls\\软件\\tensorflow-mnist", one_hot=True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()

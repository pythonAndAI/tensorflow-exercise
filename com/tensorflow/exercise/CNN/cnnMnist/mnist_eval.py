import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from com.tensorflow.exercise.CNN.cnnMnist import mnist_inference
from com.tensorflow.exercise.CNN.cnnMnist import mnist_train

#每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SIZE = 10

def evaluate(mnist):
    x = tf.placeholder(dtype=tf.float32, shape=[mnist.validation.num_examples, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name="x-input")
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.NUM_LABELS], name="y-input")

    input_tensor = np.reshape(mnist.validation.images, (mnist.validation.num_examples, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))

    validation_feed = {x: input_tensor, y_: mnist.validation.labels}

    y = mnist_inference.inference(x, False, None)

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variable_to_restore = variable_average.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)

    # 每隔EVAL_INTERVAL_SIZE秒调用一次计算正确率的过程以检测训练过程中正确率的变化
    while True:
        with tf.Session() as sess:
            # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
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

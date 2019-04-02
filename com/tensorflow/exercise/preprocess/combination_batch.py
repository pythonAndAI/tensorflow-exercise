import tensorflow as tf
import os
from com.tensorflow.exercise.preprocess import image_size as basis_code
from com.tensorflow.exercise.logging import LOG
'''
组合训练数据(batching)
将图片进行预处理之后就可以得到提供给神经网络输入层的训练数据了，在前面介绍过，将多个输入样例组织成一个batch可以提高模型训练的效率。所以在得到单个样例的预处理结果之后，
还需要将他们组织成batch，然后在提供给神经网络的输入层。Tensorflow提供了tf.train.batch和tf.train.shuffle_batch函数来将单个的样例组织成batch的形式输出。这两个函数都
会生成一个队列，队列的入队操作是生成单个样例的方法，而每次出队得到的是一个batch的样例，他们唯一的区别在于是否会将数据顺序打乱。
'''
def read_file(filepath):
    files = tf.train.match_filenames_once(os.path.join(filepath, "data.tfrecords-*"))
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    reader = tf.TFRecordReader()
    _, seriallzed_example = reader.read(filename_queue)
    example = tf.parse_single_example(seriallzed_example, features={
        "img_data" : tf.FixedLenFeature([], tf.string),
        "labels" : tf.FixedLenFeature([], tf.int64)
    })
    img_data = tf.decode_raw(example['img_data'], tf.uint8)
    labels = tf.cast(example['labels'], tf.int32)
    img_raw = []
    label_raw = []
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # for i in range(2):
            #需要转换维度
        img_raw.append(tf.reshape(img_data, [1, 30]))
        label_raw.append(tf.constant(sess.run(labels), shape=[1]))
        coord.request_stop()
        coord.join(threads)
    return img_raw, label_raw

#可以通过设置tf.train.batch和tf.train.shuffle_batch的num_threads参数，可以指定多个线程同时执行入队操作，入队操作就是数据读取以及预处理过程。当num_threads大于1时，多个线程
#会同时读取一个文件中的不同样例并进行预处理。如果需要多个线程处理不同文件中的样例时，可以使用tf.train.batch_join和tf.train.shuffle_batch_join函数。
def batch_test(filepath):
    #获取解析的数据
    img_raw, label_raw = read_file(filepath)
    #一个batch中样例的个数
    batch_size= 3
    #组合样例的队列中最多可以存储的样例个数，一般来说这个队列的大小会和每个batch的大小相关。
    capacity = 1000 + 3 * batch_size
    #使用tr.train.batch函数组合样例，[img_raw, label_raw]参数给出了需要组合的元素，一般img和label分别代表训练样本和这个样本对应的正确标签。batch_size参数给出了每个batch
    #中样例的个数，capacity给出了队列的最大容量，当队列长度等于容量时，Tensorflow将暂停入队操作，而只是等待元素出队。当元素个数小于容量时，Tensorflow将自动重新启动入队操作。
    img_batch, label_batch = tf.train.batch([img_raw, label_raw], batch_size=batch_size, capacity=capacity)
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # for i in range(6):
        LOG.getlogger("img batch").info(sess.run(img_batch))
        LOG.getlogger("img labels").info(sess.run(label_batch))
        coord.request_stop()
        coord.join(threads)
        print("all threads stop!")

def shuffle_batch_test(filepath):
    #获取解析的数据
    img_raw, label_raw = read_file(filepath)
    #一个batch中样例的个数
    batch_size= 3
    #组合样例的队列中最多可以存储的样例个数，一般来说这个队列的大小会和每个batch的大小相关。
    capacity = 1000 + 3 * batch_size
    #shuffle_batch和batch函数基本一致，但是min_after_dequeue参数是shuffle_batch函数所独有的。min_after_dequeue参数限制了出队时队列中元素的最少个数，
    # 当队列中元素太少时，随机打乱样例顺序的作用就不大了。
    img_batch, label_batch = tf.train.shuffle_batch([img_raw, label_raw], batch_size=batch_size, capacity=capacity, min_after_dequeue=30)
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # for i in range(6):
        LOG.getlogger("img batch").info(sess.run(img_batch))
        LOG.getlogger("img labels").info(sess.run(label_batch))
        coord.request_stop()
        coord.join(threads)
        print("all threads stop!")

if __name__ == "__main__":
    filepath = basis_code.get_andclean_image()
    filepath = os.path.dirname(filepath)
    # batch_test(filepath)
    shuffle_batch_test(filepath)
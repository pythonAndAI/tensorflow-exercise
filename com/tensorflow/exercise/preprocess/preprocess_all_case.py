import tensorflow as tf
import os
from com.tensorflow.exercise.preprocess import image_size as basis_code
from com.tensorflow.exercise.preprocess import preprocess_case
from com.tensorflow.exercise.mnist.final import mnist_inference
'''
下面这个案例将完成预处理全流程，也就是下面的流程:
指定原始数据的文件列表---->创建文件列表队列---->从文件中读取数据---->数据预处理---->整理成batch作为神经网络输入
'''
if __name__ == "__main__":
    filepath = os.path.dirname(basis_code.get_andclean_image())
    #1.指定原始文件列表
    files = tf.train.match_filenames_once(os.path.join(filepath, "data.tfrecords-*"))
    #2.创建文件列表队列
    filename_queue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    #3.从文件中读取数据
    _, seriallzed_example = reader.read_up_to(filename_queue, 3)
    features = tf.parse_example(seriallzed_example, features={
        "img_data" : tf.FixedLenFeature([], tf.string),
        "labels" : tf.FixedLenFeature([], tf.int64)
    })
    img_data = tf.decode_raw(features["img_data"], tf.uint8)
    labels = tf.cast(features["labels"], tf.int32)
    #定义神经网络输入层大小
    image_size = 784
    #此处必须转换为3维
    img_data = tf.reshape(img_data, [3, 10, 3])
    labels = tf.reshape(labels, [1, 3])

    #4.进行预处理
    img_data = preprocess_case.preprocess_for_train(img_data, 1, image_size, None)
    img_data = tf.reshape(img_data, [1, 784, 3])

    #5.整理成batch作为神经网络输入
    batch_size = 3
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    img_batch, label_batch = tf.train.shuffle_batch([img_data, labels], batch_size=batch_size, min_after_dequeue=min_after_dequeue, capacity=capacity)
    #定义神经网络结构
    img_batch = tf.reshape(img_batch, [9, 784])
    y = mnist_inference.inference(img_batch, None)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.8, global_step, 3, 0.99)
    label_batch = tf.reshape(label_batch, [9, 1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(label_batch, 1))
    loss = tf.reduce_mean(cross_entropy)
    train_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1000):
            sess.run(train_steps)
            if i % 100 == 0:
                print(sess.run(loss))
        coord.request_stop()
        coord.join(threads)



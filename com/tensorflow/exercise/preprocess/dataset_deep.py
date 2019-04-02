import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
from com.tensorflow.exercise.preprocess import preprocess_case
from com.tensorflow.exercise.CNN.cnnMnist import mnist_inference

#数据集整合

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#创建训练和测试的TFRecord数据
def write_train_data(type = "train"):
    mnist = input_data.read_data_sets("E:\\Alls\\软件\\tensorflow-mnist", one_hot=True, dtype=tf.uint8)
    if type == "train":
        image = mnist.train.images
        labels = mnist.train.labels
        num = mnist.train.num_examples
    elif type == "test":
        image = mnist.test.images
        labels = mnist.test.labels
        num = mnist.test.num_examples
    else:
        image = mnist.validation.images
        labels = mnist.validation.labels
        num = mnist.validation.num_examples
    #定义数据的长、宽、高，为1 * 784 * 1
    width = image.shape[1]
    height = 1
    channels = 1
    writer = tf.python_io.TFRecordWriter(os.path.join(os.getcwd(), "mnist_" + type + "_data.tfrecords"))

    for i in range(num):
        #维度为1*784
        image_raw = image[i].tostring()
        features = tf.train.Example(features=tf.train.Features(feature={
            'img_data' : _bytes_feature(image_raw),
            'label' : _int64_feature(np.argmax(labels[i])),
            'width' : _int64_feature(width),
            'height': _int64_feature(height),
            'channels': _int64_feature(channels)
        }))

        writer.write(features.SerializeToString())
    writer.close()

#解析TFRecord文件
def parser(record):
    features = tf.parse_single_example(record, features={
        'img_data': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64)
    })
    decode_image = tf.decode_raw(features['img_data'], tf.uint8)
    #转换为1 * 784 * 1
    # decode_image.set_shape([features['height'], features['width'], features['channels']])
    decode_image = tf.reshape(decode_image, [1, 784, 1])
    labels = features["label"]
    return decode_image, labels

def all_process():
    #列举输入文件，训练和测试使用不同的数据
    write_train_data()
    file_train_path = os.path.join(os.getcwd(), "mnist_train_*")
    train_file = tf.train.match_filenames_once(file_train_path)
    write_train_data("test")
    file_test_path = os.path.join(os.getcwd(), "mnist_test_*")
    test_file = tf.train.match_filenames_once(file_test_path)

    #定义神经网络的大小
    image_size = 28 #定义神经网络输入层图片的大小
    batch = 100 #定义组合数据batch的大小
    shuffer_buffer = 10000  #定义随机打乱数据时buffer的大小

    dataset = tf.data.TFRecordDataset(train_file)
    dataset = dataset.map(parser)

    #image输入为1 * 784 * 1，输出为100 * 28 *28 *1
    dataset = dataset.map(lambda image, label : (preprocess_case.preprocess_for_train(image, image_size, image_size, None), label))
    dataset = dataset.shuffle(shuffer_buffer).batch(batch)

    #重复NUM_EPOCHS个epoch，和mnist中的TRAINNG_ROUNDS指定了训练的轮数，而这里指定了整个数据集重复的次数，它也间接的确定了训练的轮数
    NUM_EPOCHS = 10
    dataset = dataset.repeat(NUM_EPOCHS)

    #定义数据迭代器，虽然定义数据集时没有直接使用placeholder来提供文件地址，但是tf,train.match_filename_once方法得到的结果和与placeholder的机制类似，
    # 也需要初始化，所以这里使用的是make_initializable_iterator
    iterator = dataset.make_initializable_iterator()
    image_batch, label_batch = iterator.get_next()
    image_batch = tf.reshape(image_batch, [100, 28, 28, 1])

    #定义神经网络的结构以及优化过程
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    y = mnist_inference.inference(image_batch, True, regularizer)

    global_step = tf.Variable(0 , trainable=False)
    #定义滑动平均模型，损失函数，指数衰减法
    variable_average = tf.train.ExponentialMovingAverage(0.99, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(0.8, global_step, 55000 / batch, 0.99)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op("all_process")

    #定义测试数据集，与训练时不同，测试数据的Dataset不需要经过随机翻转等预处理操作，也不需要打乱顺序和重复多个epoch。这里使用与训练数据相同的parser进行解析，
    # 调整分辨率到神经网络输入层大小，然后直接进行batch操作
    test_dataset = tf.data.TFRecordDataset(test_file)
    test_dataset = test_dataset.map(parser).map(lambda image, label: (tf.image.resize_images(image, [image_size, image_size]), label))
    test_dataset = test_dataset.batch(batch)

    #定义测试数据上的迭代器
    test_iterator = test_dataset.make_initializable_iterator()
    test_img_batch, test_label = test_iterator.get_next()
    test_img_batch = tf.reshape(test_img_batch, [100, 28, 28, 1])
    #定义预测结果为logit值最大的分类
    test_y = mnist_inference.inference(test_img_batch, False, None)
    prodictions = tf.argmax(test_y, -1, output_type=tf.int32)

    with tf.Session() as sess:
        #初始化变量
        sess.run((tf.local_variables_initializer(), tf.global_variables_initializer()))

        #初始化训练数据的迭代器
        sess.run(iterator.initializer)

        #循环进行训练，直到数据集完成输入、抛出OutOfRangeRrror错误
        while True:
            try:
                sess.run(train_op)
            except tf.errors.OutOfRangeError:
                print("训练数据输入完毕")
                break

        #初始化测试数据的迭代器
        sess.run(test_iterator.initializer)
        #获取预测结果
        test_result = []
        test_labels = []
        while True:
            try:
                pre, label = sess.run([prodictions, test_label])
                test_result.extend(pre)
                test_labels.extend(label)
            except tf.errors.OutOfRangeError:
                print("测试数据输入完毕")
                break

        #计算准确率
        correct = [float (y_t == y_) for (y_t, y_) in zip(test_result, test_labels)]
        accuracy = sum(correct)/ len(correct)
        print("Test accuracy is:", accuracy)


if __name__ == "__main__":
    all_process()



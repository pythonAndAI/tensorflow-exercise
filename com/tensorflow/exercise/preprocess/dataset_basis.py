import tensorflow as tf
from com.tensorflow.exercise.preprocess import image_size as basis_code
import os
import glob
'''
1.除队列之外，Tensorflow还提供了一套更高层的数据处理框架。在新的框架中，每一个数据来源被抽象成一个“数据集”，开发者可以以数据集为基本对象，方便地进行batching、随机打乱(shuffle)等操作。
2.在数据集框架中，每一个数据集代表一个数据来源：数据可能来自一个张量，一个TFRecord文件，一个文本文件，或者经过sharding的一系列文件。
从以下例子可以看出，利用数据集读取数据有三个基本步骤：
1.> 定义数据集的构造方法-->下面的例子分别从张量、文件、TFRecord文件创建数据集，对应不同的创建方法
2.> 定义遍历器-->下面介绍了两种遍历器make_one_shot_iterator和make_initializable_iterator遍历器
3.> 使用get_next方法从遍历器中读取数据张量，作为计算图其他部分的输入
'''

def tensor_dataset():
    #从一个数组中创建数据集
    input_data = [1, 2, 3, 5, 8]
    dataset = tf.data.Dataset.from_tensor_slices(input_data)

    #定义一个迭代器遍历数据集
    iterator = dataset.make_one_shot_iterator()
    #返回下一个元素
    x = iterator.get_next()
    y = x * x
    with tf.Session() as sess:
        for i in range(len(input_data)):
            print(sess.run(y))

def text_dataset():
    filepath = os.path.dirname(basis_code.get_andclean_image())
    #构造文件
    for i in range(2):
        filepaths = os.path.join(filepath, "input_files" + str(i))
        file = open(filepaths, "a+")
        for j in range(5):
            if i == 1:
                j = j + 5
            file.write(str(j))
        file.close()
    #获取构造的文件路径列表
    files = glob.glob(os.path.join(filepath, "input_files*"))
    #创建数据集
    dataset = tf.data.TextLineDataset(files)
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    with tf.Session() as sess:
        for i in range(2):
            print(sess.run(x))
    #删除构造的文件
    for file in files:
        os.remove(file)

def parse(record):
    features = tf.parse_single_example(record, features={
        'pixels': tf.FixedLenFeature([], tf.int64),
        'labels': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    #转换类型，把tf.int64转换为tf.int32
    pixels = tf.cast(features['pixels'], tf.int32)
    labels = tf.cast(features['labels'], tf.int32)

    return image, pixels, labels

def tfrecord_dataset():
    filepath = os.path.dirname(basis_code.get_andclean_image())
    #从TFRecord文件创建数据集
    files = glob.glob(os.path.join(filepath, "output.tfrecords*"))
    dataset = tf.data.TFRecordDataset(files)
    #map()函数表示对数据集中的每一条数据进行调用相应的方法。使用TFRecordDataset读取的是二进制的数据，这里需要通过map()来调用parse对二进制数据进行解析。
    dataset = dataset.map(parse)
    #定义迭代器
    iterator = dataset.make_one_shot_iterator()
    image, pixels, labels = iterator.get_next()
    with tf.Session() as sess:
        for i in range(10):
            print(sess.run([image, pixels, labels]))

#以上例子使用了最简单的make_one_shot_iterator来遍历数据集，在使用make_one_shot_iterator时，数据集的所有参数必须已经确定，因此make_one_shot_iterator不需要特别的初始化过程。
#如果需要用placeholder来初始化数据集，那就需要用到make_initializable_iterator。如下
def initializable_iterator():
    filepath = os.path.dirname(basis_code.get_andclean_image())
    files = glob.glob(os.path.join(filepath, "output.tfr*"))
    #从TFRecord文件创建数据集，具体文件路径是一个placeholder，稍后再提供具体的路径
    place_file = tf.placeholder(tf.string)
    dataset = tf.data.TFRecordDataset(place_file)
    dataset = dataset.map(parse)
    #还有其他迭代器，比如reinitializable_iterator和feedable_iterator两种更加灵活的迭代器，前者可以多次initialize用于遍历不同的数据来源，而后者可以用feed_dict的方式动态指定运行哪个iterator.具体使用参考官网API
    iterator = dataset.make_initializable_iterator()
    image, pixels, labels = iterator.get_next()
    with tf.Session() as sess:
        #首先初始化迭代器，并给出文件路径
        sess.run(iterator.initializer, feed_dict={place_file : files})
        #遍历所有数据，当遍历结束时，程序会抛出OutOfRangeError。在实际问题中，我们也不太清楚应该遍历几次，所以可以用while循环。
        while True:
            try:
                print(sess.run([image, pixels, labels]))
            except tf.errors.OutOfRangeError:
                break

if __name__ == "__main__":
    # tensor_dataset()
    # text_dataset()
    # tfrecord_dataset()
    initializable_iterator()
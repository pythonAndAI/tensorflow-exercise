# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
from com.tensorflow.exercise.logging import LOG

'''
在preprocess.py中我们是通过一个字典来存储图片信息，但是当数据来源更加复杂，每一个样例中的信息更加丰富之后，这种方式就很难有效地记录输入数据的信息了。
于是tensorflow提供了TFRecord的格式来统一存储数据。
TFRecord文件中的数据都是通过tf.train.Example Protocol Buffer的格式来存储的。以下代码给出了tf.train.Example的定义
message Example {
  Features features = 1;
};

message Features {
  map<String, Feature> feature = 1;
};

message Feature {
  oneof kind {
    BytesList byte_list = 1; #字符串
    FloatList float_list = 1; #实数列表
    Int64List int64_list = 1; #整数列表
  }
};
以上可以看出tf.train.Example的数据结构是比较整洁的。tf.train.Example中包含了一个从属性名称到取值的字典。其中属性名称为一个字符串，属性取值可以为字符串(BytesList)、
实数列表(FloatList)、整数列表(Int64List)。比如将一张解码前的图像存为一个字符串，图像所对应的类别编号存为整数列表
'''
# 输出TFRecord文件的地址
FILENAME = os.path.join(os.path.dirname(os.getcwd()), "preprocess/output.tfrecords")
#生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_TFRecord():
    #此处dtype只能是uint8和flaot32
    mnist = input_data.read_data_sets("E:\\Alls\\软件\\tensorflow-mnist", one_hot=True, dtype=tf.uint8)

    images = mnist.train.images
    labels = mnist.train.labels
    num_example = mnist.train.num_examples
    #输出为(55000, 784),取第一个下标为784
    pixels = images.shape[1]

    #创建一个write来写TFRecord文件
    write = tf.python_io.TFRecordWriter(FILENAME)
    for i in range(num_example):
        image_raw = images[i].tostring()
        #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构
        #注意features=tf.train.Features(feature这块的单词书写，一个字母都不能错
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'pixels' : _int64_feature(pixels),
                    'labels' : _int64_feature(np.argmax(labels[i])),
                    'image_raw' : _bytes_feature(image_raw)
                }
            )
        )
        try:
            write.write(example.SerializeToString())
        except IOError:
            print("writer error!")
            write.close()
    write.close()

def read_TFRercod():
    #创建一个reader来获取TFRecord文件的样例
    reader = tf.TFRecordReader()
    #创建一个队列来维护输入文件列表
    fileName_queue = tf.train.string_input_producer([FILENAME])
    #从文件中读取一个样例。也可以使用read_up_to函数一次性读取多个样例
    _, serialized_example = reader.read(fileName_queue)
    #解析读入的一个样例，如果需要解析多个样例，可以用parse_example
    features = tf.parse_single_example(serialized_example, features={
        #必须和写入时的类型一致
        'pixels': tf.FixedLenFeature([], tf.int64),
        'labels': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })

    #tf.decode_raw可以将字符串解析成图像对应的像素数组.必须和读取mnist数据集时设置的类型一致
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    #转换类型，把tf.int64转换为tf.int32
    pixels = tf.cast(features['pixels'], tf.int32)
    labels = tf.cast(features['labels'], tf.int32)

    sess = tf.Session()
    #启动多线程处理输入数据
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #每次运行可以读取TFRecord文件中的一个样例
    # for i in range(10):
    LOG.getlogger("read").info(sess.run([image, pixels, labels]))
    pass

if __name__ == "__main__":
    write_TFRecord()
    read_TFRercod()
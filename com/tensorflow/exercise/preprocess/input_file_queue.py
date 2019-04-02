import tensorflow as tf
from com.tensorflow.exercise.preprocess import image_size as basis_code
import os
from com.tensorflow.exercise.logging import LOG
'''
输入文件队列
1.虽然一个TFRecord文件中可以存储多个训练样例。但是当训练数据量较大时，可以将数据分成多个TFRecord文件来提高处理效率。
2.Tensorflow提供了tf.train.match_filenames_once函数来获取符合一个正则表达式的所有文件，得到的文件列表可以通过tf.train.string_input_producer函数进行有效地管理。
3.tf.train.string_input_producer函数会使用初始化时提供的文件列表创建一个输入队列，输入队列中原始的元素为文件列表中的所有文件。创建好的输入队列可以作为文件读取函数的参数。
每次调用文件读取函数时，该函数会先判断当前是否已有打开的文件可读，如果没有或者打开的文件已经读完，这个函数会从输入队列中出队一个文件并从这个文件中读取。
4.通过设置shuffle参数，tf.train.string_input_producer函数支持随机打乱文件列表中文件出队的顺序，当shuffle参数为True时，文件在加入队列之前会被打乱顺序，
所以出队的顺序也是随机地。tf.train.string_input_producer生成的输入队列可以同时被多个文件读取线程操作，而且输入队列会将队列中的文件均匀地分给不同的线程，
不出现有些文件被处理过多次而有些文件还没有被处理的情况。
5.当一个输入队列中的所有文件都被处理完后，它会将初始化时提供的文件列表中的文件重新加入队列。tf.train.string_input_producer函数可以设置num_epochs参数来限制加载初始文件列表的最大轮数。
在测试神经网络模型时，因为所有测试数据只需要使用一次，所以可以将num_epochs参数设置为1。
'''
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecord(decode_path, write_path):
    img_data = basis_code.basis_decode(decode_path)
    #维度为6x10x3
    img_data = tf.image.resize_images(img_data, [6, 10], method=1)
    with tf.Session() as sess:
        img_value = sess.run(img_data)
    #总共写入多少文件
    num_shards = 2
    #每个文件写入多少数据
    instance_per_shard = 3
    for i in range(2):
        #将文件分为多个文件。以便读取时方便匹配
        filename = os.path.join(write_path, "data.tfrecords-" + str(i) + ".5d-of-" + str(num_shards) + ".5d")
        writer = tf.python_io.TFRecordWriter(filename)
        for j in range(instance_per_shard):
            if i == 1:
                j = j + 3
            LOG.getlogger("write img" + str(j)).info(img_value[j])
            img_raw = img_value[j].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                "labels": _int64_feature(j),
                "img_data" : _byte_feature(img_raw)
            }))
            try:
                writer.write(example.SerializeToString())
            except IOError:
                print("write error!")
                writer.close()
        writer.close()

def read_tfrecord(filepath):
    files = tf.train.match_filenames_once(os.path.join(filepath, "data.tfrecords-*"))
    #通过tf.train.string_input_producer函数创建输入队列，输入队列中的文件列表为tf.train.match_filenames_once函数获取的文件列表。这里将shuffle参数设置为False.
    #来避免随机打乱读文件的顺序，但一般在解决真实问题时，会将shuffle参数设置为True
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    reader = tf.TFRecordReader()
    _, seriallzed_example = reader.read(filename_queue)
    #img_data和labels要和写入的名字保持一致
    example = tf.parse_single_example(seriallzed_example, features={
        "img_data" : tf.FixedLenFeature([], tf.string),
        "labels" : tf.FixedLenFeature([], tf.int64)
    })

    image = tf.decode_raw(example["img_data"], tf.uint8)
    labels = tf.cast(example["labels"], tf.int32)

    with tf.Session() as sess:
        #虽然在本段程序中没有声明任何变量，但使用tf.train.match_filenames_once函数时需要初始化一些变量
        tf.local_variables_initializer().run()
        LOG.getlogger("file names").info(sess.run(files))
        #声明tf.train.Coordinator类来协同不同线程，并启动线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(6):
            LOG.getlogger("read img data").info(sess.run(image))
            LOG.getlogger("read labels").info(sess.run(labels))

        coord.request_stop()
        coord.join(threads)
        print("all thread stop!")


if __name__ == "__main__":
    decode_path = basis_code.get_andclean_image()
    write_path = os.path.dirname(decode_path)
    write_tfrecord(decode_path, write_path)
    read_tfrecord(write_path)

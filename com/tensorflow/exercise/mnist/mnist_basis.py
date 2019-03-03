import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#mnist有训练、测试、验证数据，把输入值像素矩阵的值控制在[0, 1]之间，代表了颜色的深浅。其中0代表白色背景，1代表黑色背景
def mnist_download():
    path = "E:\\Alls\\软件\\tensorflow-mnist";
    #如果path下没有下载好的数据集，则会自动下载mnist数据集到path目录
    mnist = input_data.read_data_sets(path, one_hot=True)

    #打印Training data size：55000
    print("Training data size:", mnist.train.num_examples)

    # validation data size：5000
    print("validation data size:", mnist.validation.num_examples)

    # 打印Test data size：10000
    print("Test data size:", mnist.test.num_examples)

    # 打印Example training size：[0 0, 0, ... 0.380 0.376]
    print("Example training data:", mnist.train.images[0])

    # 打印Example training data label：55000
    #[[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
    print("Example training data label:", mnist.train.labels[0])

    batch_size = 100
    #为了方便使用随机梯度下降，input_data.read_data_sets函数生成的类还提供了train.next_batch函数，它可以从所有的训练数据中读取一小部分作为一个训练的batch。如下
    #xs shape is: (100, 784)   ys shape is: (100, 10)
    xs, ys = mnist.train.next_batch(batch_size)
    print("xs shape is:", xs.shape)
    print("ys shape is:", ys.shape)

if __name__  == "__main__":
    mnist_download()
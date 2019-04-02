import tensorflow as tf
from com.tensorflow.exercise.CNN.cnnMnist import mnist_basis as basis
from com.tensorflow.exercise.preprocess import image_size as basis_code
import numpy as np

'''
一般只会对训练数据进行预处理，测试数据和验证数据一般不需要进行预处理
在解决真实的图像识别问题时，一般会同时使用多种处理方法。下面展示如何将不同的图像处理函数结合成一个完成的图像预处理流程。以下程序完成了从图像片段截取，
到图像大小调整再到图像翻转以及色彩调整的整个图像预处理过程
'''
#给定一张图像，随机调整图像的色彩，因为调整亮度、对比度、饱和度和色相的顺序会影响最后得到的结果，所以可以定义多种不同的顺序。具体使用哪一种顺序可以在训练数据预处理时随机
#地选择一种。这样可以进一步降低无关因素对模型的影响
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    #还可以定义其他的顺序
    return tf.clip_by_value(image, 0.0, 1.0)

#给定一张解码后的图像、目标图像的尺寸以及图像上的标注框，此函数可以对给出的图像进行预处理，这个函数的输入图像是图像识别问题中原始的训练图像，而输出则是神经
#网络模型的输入层，注意这里只处理模型的训练数据，对于测试数据，一般不需要使用随机变量的步骤。
def preprocess_for_train(image, height, width, bbox):
    #如果没有提供标注框，则认为整个图像就是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    #转换数据类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, tf.float32)
    #随机截取图像，减少需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    #将随机截取的图像调整为神经网络输入层的大小。大小调整算法是随机选择的
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
    #随机上下翻转图像
    distorted_image = tf.image.random_flip_up_down(distorted_image)
    #随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    #随机使用一种顺序调整图片色彩
    distorted_image = distort_color(distorted_image, color_ordering=np.random.randint(2))
    return distorted_image

#通过上面程序，就可以通过一张训练图像衍生出很多训练样本。通过将训练图像预处理，训练得到的神经网络模型可以识别不同大小、方位、色彩等方面的实体
if __name__ == "__main__":
    filepath = basis_code.get_andclean_image()
    image = basis_code.basis_decode(filepath)
    boxes = tf.constant([[[0.13, 0.24, 0.55, 0.89], [0.33, 0.43, 0.48, 0.67]]])
    with tf.Session() as sess:
        #运行6次获得6种不同的图像
        for i in range(6):
            result = preprocess_for_train(image, 180, 267, boxes)
            basis.drawing(sess.run(result))
            basis_code.basis_encode(result, basis_code.get_encode_path(filepath, i))


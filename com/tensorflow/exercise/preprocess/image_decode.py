import tensorflow as tf
from com.tensorflow.exercise.logging import LOG
from com.tensorflow.exercise.CNN.cnnMnist import mnist_basis as basis

#图像的编码和解码
if __name__ == "__main__":
    #读取图像的原始数据
    image_raw_data = tf.gfile.FastGFile("E:\\Alls\\code\\psb.jpg", 'rb').read()
    with tf.Session() as sess:
        #对图像进行jpeg的格式解码从而得到图像对应的三维矩阵。Tensorflow还提供了tf.image.decode_png格式的图像进行解码。解码之后的结果为一个张量
        #在使用它的取值之前需要明确调用运行的过程
        img_data = tf.image.decode_jpeg(image_raw_data)
        LOG.getlogger("image decode data").info(sess.run(img_data))
        LOG.getlogger("image decode shape").info(sess.run(img_data).shape)
        basis.drawing(sess.run(img_data))

        #将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件中。编码时img_data的类型必须是uint8
        encode_image = tf.image.encode_jpeg(img_data)
        with tf.gfile.FastGFile("E:\\Alls\\code\\psb1.jpg", 'wb') as f:
            LOG.getlogger("image encode shape").info(encode_image)
            f.write(sess.run(encode_image))





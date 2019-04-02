import tensorflow as tf
from com.tensorflow.exercise.CNN.cnnMnist import mnist_basis as basis
from com.tensorflow.exercise.preprocess import image_size as basis_code

#色彩调整，调整图像的亮度、对比度、饱和度和色相在很多图像识别应用中都不会影响识别结果

#亮度调整
def adjust_brightness(filepath):
    img_data = basis_code.basis_decode(filepath)
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    #将图像的亮度-0.5
    adjust_less = tf.image.adjust_brightness(img_data, -0.5)
    #色彩调整的API可能导致像素的实数值超出0.0-1.0的范围，因此在输出最终图像前需要将其值截断到0.0-1.0范围之间，否则不仅图像无法正常可视化，以此为输入的神经网络的训练质量也可能受到影响。
    #如果对图像进行多项处理操作，那么这一截断过程应当在所有处理完成后进行。
    adjust_less = tf.clip_by_value(adjust_less, 0.0, 1.0)

    adjust_add = tf.image.adjust_brightness(img_data, 0.5)
    adjust_add = tf.clip_by_value(adjust_add, 0.0, 1.0)
    #在[-0.7, 0.7)的范围内随机调整图像的亮度
    random_adjust = tf.image.random_brightness(img_data, 0.7)
    random_adjust = tf.clip_by_value(random_adjust, 0.0, 1.0)

    with tf.Session() as sess:
        basis.drawing(sess.run(adjust_less))
        basis.drawing(sess.run(adjust_add))
        basis.drawing(sess.run(random_adjust))
    basis_code.basis_encode(adjust_less, basis_code.get_encode_path(filepath, 1))
    basis_code.basis_encode(adjust_add, basis_code.get_encode_path(filepath, 2))
    basis_code.basis_encode(random_adjust, basis_code.get_encode_path(filepath, 3))

#对比度调整
def adjust_contrast(filepath):
    img_data = basis_code.basis_decode(filepath)
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    #将图像的对比度减少0.5倍
    adjust_less = tf.image.adjust_contrast(img_data, 0.5)
    # 将图像的对比度增加5倍
    adjust_add = tf.image.adjust_contrast(img_data, 5)
    #在[0.3, 1)的范围内随机调整图像的对比度
    random_adjust = tf.image.random_contrast(img_data, 0.3, 1)

    with tf.Session() as sess:
        basis.drawing(sess.run(adjust_less))
        basis.drawing(sess.run(adjust_add))
        basis.drawing(sess.run(random_adjust))
    basis_code.basis_encode(adjust_less, basis_code.get_encode_path(filepath, 1))
    basis_code.basis_encode(adjust_add, basis_code.get_encode_path(filepath, 2))
    basis_code.basis_encode(random_adjust, basis_code.get_encode_path(filepath, 3))

#色相调整
def adjust_hue(filepath):
    img_data = basis_code.basis_decode(filepath)
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    #下列依次将图像的色相加0.1、0.3、0.6、0.9
    adjust_add1 = tf.image.adjust_hue(img_data, 0.1)
    adjust_add2 = tf.image.adjust_hue(img_data, 0.3)
    adjust_add3 = tf.image.adjust_hue(img_data, 0.6)
    adjust_add4 = tf.image.adjust_hue(img_data, 0.9)
    #在[0.3, 1)的范围内随机调整图像的色相
    random_adjust = tf.image.random_hue(img_data, 0.3, 1)

    with tf.Session() as sess:
        basis.drawing(sess.run(adjust_add1))
        basis.drawing(sess.run(adjust_add2))
        basis.drawing(sess.run(adjust_add3))
        basis.drawing(sess.run(adjust_add4))
        basis.drawing(sess.run(random_adjust))
    basis_code.basis_encode(adjust_add1, basis_code.get_encode_path(filepath, 1))
    basis_code.basis_encode(adjust_add2, basis_code.get_encode_path(filepath, 2))
    basis_code.basis_encode(adjust_add3, basis_code.get_encode_path(filepath, 3))
    basis_code.basis_encode(adjust_add4, basis_code.get_encode_path(filepath, 4))
    basis_code.basis_encode(random_adjust, basis_code.get_encode_path(filepath, 5))

#饱和度调整
def adjust_saturation(filepath):
    img_data = basis_code.basis_decode(filepath)
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    #将图像的饱和度-5
    adjust_less = tf.image.adjust_saturation(img_data, -5)
    # 将图像的饱和度+5
    adjust_add = tf.image.adjust_saturation(img_data, 5)
    #在[-9, 3)的范围内随机调整图像的饱和度
    random_adjust = tf.image.random_saturation(img_data, 1, 4)

    with tf.Session() as sess:
        basis.drawing(sess.run(adjust_less))
        basis.drawing(sess.run(adjust_add))
        basis.drawing(sess.run(random_adjust))
    basis_code.basis_encode(adjust_less, basis_code.get_encode_path(filepath, 1))
    basis_code.basis_encode(adjust_add, basis_code.get_encode_path(filepath, 2))
    basis_code.basis_encode(random_adjust, basis_code.get_encode_path(filepath, 3))

#除了调整图像的亮度、对比度、饱和度和色相，还可以完成图像标准化的过程。这个操作就是将图像上的亮度均值变为0，方差变为1
def adjust_standardization(filepath):
    img_data = basis_code.basis_decode(filepath)
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    #将图像的亮度均值变为0，方差变为1
    adjust = tf.image.per_image_standardization(img_data)

    with tf.Session() as sess:
        basis.drawing(sess.run(adjust))
    basis_code.basis_encode(adjust, basis_code.get_encode_path(filepath, 1))

if __name__ == "__main__":
    filepath = basis_code.get_andclean_image()
    # adjust_brightness(filepath)
    # adjust_contrast(filepath)
    # adjust_hue(filepath)
    # adjust_saturation(filepath)
    adjust_standardization(filepath)

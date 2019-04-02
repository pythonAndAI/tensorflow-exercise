import tensorflow as tf
from com.tensorflow.exercise.CNN.cnnMnist import mnist_basis as basis
from com.tensorflow.exercise.preprocess import image_size as basis_code

#通过tf.image.draw_bounding_boxes函数在图像中加入标注框
def add_box(filepath):
    img_data = basis_code.basis_decode(filepath)
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    #将图像缩小一点，这样可视化能让标注框更加清楚
    img_data = tf.image.resize_images(img_data, [180, 267], method=1)
    #tf.image.draw_bounding_boxes函数要求图像矩阵中的数字为实数，上面已经转换过。而且tf.image.draw_bounding_boxes函数的输入是一个batch的数据。
    #也就是多张图像组成的四维矩阵，所以需要将解码之后的图像矩阵加一维
    img_data = tf.expand_dims(img_data, 0)
    #下面定义表示有两个标注框。一个标注框有4个数字，分别代表[Ymin, Xmin, Ymax, Xmax]。这里的数字都是图像的相对位置，比如在180x267的图像中。
    #[0.33, 0.43, 0.48, 0.67]代表的大小为[0.33*180, 0.43*267, 0.48*180, 0.67*237]
    boxes = tf.constant([[[0.13, 0.24, 0.55, 0.89], [0.33, 0.43, 0.48, 0.67]]])
    result_data = tf.image.draw_bounding_boxes(img_data, boxes)
    with tf.Session() as sess:
        basis.drawing(sess.run(result_data[0]))
    basis_code.basis_encode(result_data[0], basis_code.get_encode_path(filepath, 1))

#截取选中的标注框
def slice_box(filepath):
    img_data = basis_code.basis_decode(filepath)
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    #调整大小
    img_data = tf.image.resize_images(img_data, [180, 267], method=1)
    #定义标注框大小
    boxes = tf.constant([[[0.13, 0.24, 0.55, 0.89], [0.33, 0.43, 0.48, 0.67]]])
    #可以通过提供标注框的方式来告诉随机截取图像的算法那些部分是“有信息量”的.min_object_covered=0.4表示截取部分至少包含某个标注框的40%内容
    #bbox_for_draw为重新随机返回的一个标注框大小
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(img_data), bounding_boxes=boxes, min_object_covered=0.4)
    #增加一个维度
    batch = tf.expand_dims(img_data, 0)
    box_data = tf.image.draw_bounding_boxes(batch, bbox_for_draw)
    #随机截取出来的图像。因为算法带有随机成分，所以每次得到的结果会有所不同
    distorted_image = tf.slice(img_data, begin, size)
    with tf.Session() as sess:
        basis.drawing(sess.run(img_data))
        basis.drawing(sess.run(box_data[0]))
        basis.drawing(sess.run(distorted_image))
    basis_code.basis_encode(img_data, basis_code.get_encode_path(filepath, 1))
    basis_code.basis_encode(box_data[0], basis_code.get_encode_path(filepath, 2))
    basis_code.basis_encode(distorted_image, basis_code.get_encode_path(filepath, 3))



if __name__ == "__main__":
    filepath = basis_code.get_andclean_image()
    # add_box(filepath)
    slice_box(filepath)

import tensorflow as tf
from com.tensorflow.exercise.CNN.cnnMnist import mnist_basis as basis
from com.tensorflow.exercise.preprocess import image_size as basis_code

#图像翻转
def method_one(filepath):
    img_data = basis_code.basis_decode(filepath)
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    #上下翻转
    up_down_data = tf.image.flip_up_down(img_data)
    #左右翻转
    left_right_data = tf.image.flip_left_right(img_data)
    #沿对角线翻转
    transposed = tf.image.transpose_image(img_data)
    with tf.Session() as sess:
        basis.drawing(sess.run(up_down_data))
        basis.drawing(sess.run(left_right_data))
        basis.drawing(sess.run(transposed))
    basis_code.basis_encode(up_down_data, basis_code.get_encode_path(filepath, 1))
    basis_code.basis_encode(left_right_data, basis_code.get_encode_path(filepath, 2))
    basis_code.basis_encode(transposed, basis_code.get_encode_path(filepath, 3))

#图像的翻转不会影响识别的效果。于是在训练图像识别的神经网络模型时，可以随机地翻转训练图像，这样训练得到的模型可以识别不同角度的实体。
#比如假设在训练数据中所有的猫头都是向右的，那么训练出来的模型就无法很好地识别猫头向左的猫。可以通过随机翻转训练图像的方式可以在0成本的情况下很大程度地缓解该问题
def method_two(filepath):
    img_data = basis_code.basis_decode(filepath)
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    #以50%概率上下翻转图像
    random_up_dowm_data = tf.image.random_flip_up_down(img_data)
    # 以50%概率左右翻转图像
    random_left_rigth = tf.image.random_flip_left_right(img_data)
    with tf.Session() as sess:
        basis.drawing(sess.run(random_up_dowm_data))
        basis.drawing(sess.run(random_left_rigth))
    basis_code.basis_encode(random_up_dowm_data, basis_code.get_encode_path(filepath, 1))
    basis_code.basis_encode(random_left_rigth, basis_code.get_encode_path(filepath, 2))


if __name__ == "__main__":
    filepath = basis_code.get_andclean_image()
    # method_one(filepath)
    method_two(filepath)

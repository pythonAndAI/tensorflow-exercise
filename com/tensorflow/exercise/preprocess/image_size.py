import tensorflow as tf
from com.tensorflow.exercise.CNN.cnnMnist import mnist_basis as basis
from com.utils import Log_Util
import os
import re

#图片大小调整

#判断字符串中是否有包含数字。包含返回True，不包含返回False
def check_num(str):
    pattern = re.compile('[0-9]+')
    for s in str:
        match = pattern.findall(s)
        if match:
            return match
    return False

#获取解码图片路径，并清理其他图片
def get_andclean_image():
    image_path = os.getcwd()
    file_name = ""
    for filename in os.listdir(image_path):
        if os.path.isdir(os.path.join(image_path, filename)):
            continue
        if len(filename.split(".")) == 1 or filename.split(".")[1] != "jpg":
            continue
        if check_num(filename):
            os.remove(os.path.join(image_path, filename))
        else:
            file_name = filename
    return os.path.join(image_path, file_name)

#获取编码路径
def get_encode_path(decodepath, num):
    basename = os.path.basename(decodepath)
    dirname = os.path.dirname(decodepath)
    base_split = basename.split(".")
    return os.path.join(dirname, base_split[0] + str(num) + "." + base_split[1])

#解码
def basis_decode(filepath):
    image_raw_data = tf.gfile.FastGFile(filepath, 'rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        basis.drawing(sess.run(img_data))
        # img_value = sess.run(img_data)
    return img_data

#编码
def basis_encode(img_data, filepath):
    if img_data.dtype != tf.uint8:
        #编码时转换类型为uint8
        img_data = tf.image.convert_image_dtype(img_data, tf.uint8)
    image_raw_data = tf.image.encode_jpeg(img_data)
    with tf.gfile.FastGFile(filepath, 'wb') as f:
        with tf.Session() as sess:
            f.write(image_raw_data.eval())


'''
第一种是通过算法使得新的图像尽量保存原始图像上的所有信息
tf.reshape是转换之前和转换之后张量的值的个数是相同的，用于CNN全连接层转换维度
tf.image.resize_images转换之前的值的个数和转换之后值的个数不一定相同

method取值                图像大小的调整算法
0                    双线性插值法
1                    最近邻居法
2                    双三次插值法
3                    面积插值法
'''
def method_one(filepath):
    before_img_data = basis_decode(filepath)
    if before_img_data.dtype != tf.float32:
        #这里tf.image.decode_png 得到的是uint8格式，范围在0-255之间，经过convert_image_dtype 就会被转换为区间在0-1之间的float32格式
        #大多数API支持整数和实数的类型的输入。如果输入时整数类型，这些API会在内部将输入转化为实数后处理，再将输出转化为整数。
        #如果有多个处理步骤，在整数和实数之间的反复转化将导致精度损失，因此推荐在图像处理前将其转化为实数类型。
        before_img_data = tf.image.convert_image_dtype(before_img_data, tf.float32)
    for i in range(4):
        #第一个参数为原始图像。第二个参数为调整后的图像大小，method参数给出了调整图像大小的算法，注意，如果输入数据为unit8格式，
        # 那么输出将是0~255之间的实数，不方便后续处理，所以建议在调整图像大小之前先转化为实数类型，用tf.image.convert_image_dtype函数
        img_data = tf.image.resize_images(before_img_data, [300, 300], method=i)
        with tf.Session() as sess:
            #原始的图像维度为(1105, 681, 3)
            #画出改变大小的图像
            basis.drawing(sess.run(img_data))
            #获取编码路径
            endoce_path = get_encode_path(filepath, i)
            Log_Util.getlogger("编码路径为:").info(endoce_path)
            #编码到新的图片中
            basis_encode(img_data, endoce_path)

#tf.image.resize_image_with_crop_or_pad对图像进行裁剪或者填充。函数的第一个参数为原始图像，后面两个参数是调整后的目标图像大小。
#如果原始图像的尺寸大于目标图像，那么这个函数会自动截取原始图像居中的部分(比如下面的croped)。
#如果原始图像的尺寸小于目标图像，那么这个函数会自动在原始图像的四周填充全0背景(比如下面的padded)。
def method_two(filepath):
    img_data = basis_decode(filepath)
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 800, 500)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 1500, 1000)
    with tf.Session() as sess:
        basis.drawing(sess.run(croped))
        basis.drawing(sess.run(padded))
    basis_encode(croped, get_encode_path(filepath, 1))
    basis_encode(padded, get_encode_path(filepath, 2))

#还支持通过比例调整图片大小
def method_three(filepath):
    img_data = basis_decode(filepath)
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)

    central_cropped = tf.image.central_crop(img_data, 0.5)
    with tf.Session() as sess:
        basis.drawing(sess.run(central_cropped))
    basis_encode(central_cropped, get_encode_path(filepath, 1))


if __name__ == "__main__":
    filepath = get_andclean_image()
    Log_Util.getlogger("原始图片路径为:").info(filepath)
    # method_one(filepath)
    # method_two(filepath)
    method_three(filepath)
    #还有其他的填充或者裁剪给定区域的图像，比如：tf.image.crop_to_bounding_box和tf.image.pad_to_bounding_box，这两个函数要求给出的尺寸满足一定的要求

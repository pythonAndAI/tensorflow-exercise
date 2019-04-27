import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from com.utils import constant_Util
from com.utils import File_Util
import matplotlib.pyplot as plt

'''
高维向量可视化。生成PROJECTOR所需要的这两个文件。
运行以下程序可以得到两个文件，一个是MNIST测试数据sprite图像，这个图像包含了所有的MNIST测试数据图像。另一个是mnist_meta.tsv，这个文件的第一行是每一列的说明，
以后的每一行代表一张图片，在这个文件中给出了每一张图对应的真实标签。 
'''
#PROJECTOR需要的日志文件名和地址相关参数
LOG_DIR = os.getcwd()

#使用给出的MNIST图片列表生成sprite图像
def get_sprite_image(images):
    if not isinstance(images, list):
        images = np.array(images)
        img_h = images.shape[1]
        img_w = images.shape[2]
        #sprite图像可以理解成是所有小图片拼成的大正方形矩阵，大正方形矩阵中的每一个元素就是原来的小图片。于是这个正方形的边长就是sqrt(n)---->平方根，因为是正方形，
        # 所以一个边就是数量的平方根。其中n为小图片的数量
        m = int(np.ceil(np.sqrt(images.shape[0])))

        #使用全1来初始化最终的大图片
        sprite_image = np.ones((img_h*m, img_w*m))

        for i in range(m):
            for j in range(m):
                #计算当前图片的编号
                cur = i * m + j
                if cur < images.shape[0]:
                    #将当前小图片的内容复制到最终的sprite图像
                    sprite_image[i*img_h:(i+1)*img_h, j*img_w:(j+1)*img_w] = images[cur]
        return sprite_image

def create_sprite_image():
    #指定one_hot=False，于是得到的labels就是一个数字，表示当前图片所表示的数字
    mnist = input_data.read_data_sets(constant_Util.MNIST_PATH, one_hot=False)
    #生成sprite图像
    to_visualise = 1 - np.reshape(mnist.test.images, (-1, 28, 28))
    sprite_image = get_sprite_image(to_visualise)

    #将生成的sprite图像放到相应的日志目录下
    path_for_mnist_sprites = os.path.join(LOG_DIR, constant_Util.SPRITE_FILE)
    plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')

    #生成每张图片对应的标签文件并写道相应的日志目录下
    path_for_mnist_metadata = os.path.join(LOG_DIR, constant_Util.META_FILE)
    with open(path_for_mnist_metadata, "w") as f:
        f.write("Index\tLabel\n")
        for index,label in enumerate(mnist.test.labels):
            f.write("%d\t%d\n" % (index,label))

if __name__ == "__main__":
    File_Util.remove_designation_file(os.path.join(LOG_DIR, constant_Util.SPRITE_FILE))
    File_Util.remove_designation_file(os.path.join(LOG_DIR, constant_Util.META_FILE))
    create_sprite_image()






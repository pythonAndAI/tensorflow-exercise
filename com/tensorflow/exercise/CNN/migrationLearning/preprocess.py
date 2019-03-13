import tensorflow as tf
import numpy as np
import os.path
import glob
from com.tensorflow.exercise.logging import LOG

#原始输入数据的目录，这个目录底下有5个子目录，每个子目录底下保存属于该类别的所有图片
INPUT_PATH = "E:\\Alls\\software\\flower_photo\\flower_photos"
#输出文件地址。将整理后的图片数据通过numpy的格式保存。
OUTPUT_FILE = "E:\\Alls\\software\\flower_photo\\flower_processed_data.npy"

#测试数据和验证数据比例
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

#读取数据并将数据分割成训练数据、验证数据和测试数据

def create_image_lists(sess=None, testing_percentage=None, validation_percentage=None):
    #获取一个元组，第一位是文件夹路径，第二位是一个list集合，里面是文件夹底下所有图片的名称
    sub_dirs = [x[0] for x in os.walk(INPUT_PATH)]
    # LOG.getlogger("sub_dirs").info(sub_dirs)

    #初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0
    #遍历获取的文件路径列表
    for sub_dir in sub_dirs:
        #获取一个子目录中所有的图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        #得到文件夹名称
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            #拼接路径。比如：E:\Alls\software\flower_photo\flower_photos\daisy\.jpg
            file_glob = os.path.join(INPUT_PATH, dir_name, '*.' + extension)
            #获取file_glob路径底下所有的图片的文件名。并赋给file_list
            file_list.extend(glob.glob(file_glob))
            #如果file_list为空，则继续循环
            if not file_list: continue

            #处理图片数据
            for file_name in file_list:
                #读取并解析图片，将图片转化为299*299以便inception-v3模型来处理
                image_raw_data = tf.gfile.FastGFile(file_name, 'rb').read()
                image = tf.image.decode_jpeg(image_raw_data)
                if image.dtype != tf.float32:
                    image = tf.image.convert_image_dtype(image, tf.float32)
                image = tf.image.resize_images(image, [299, 299])
                image_value = sess.run(image)

                #随机划分数据集
                change = np.random.randint(100)
                if change < validation_percentage:
                    validation_images.append(image_value)
                    validation_labels.append(current_label)
                elif change < (testing_percentage + validation_percentage):
                    testing_images.append(image_value)
                    testing_labels.append(current_label)
                else:
                    training_images.append(image_value)
                    training_labels.append(current_label)

                current_label += 1

    #将训练数据随机打乱以获取得更好的训练效果
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray([training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels])

def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
        #通过numpy格式保存后处理的数据
        np.save(OUTPUT_FILE, processed_data)

if __name__ == "__main__":
    # create_imag e_lists()
    main()
    pass


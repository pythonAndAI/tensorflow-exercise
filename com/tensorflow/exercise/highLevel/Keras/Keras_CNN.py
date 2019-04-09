import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import backend as k

'''
通过keras在Mnist数据集上实现LeNet5模型
从以下代码可以看出使用Keras API训练模型可以先定义一个Sequential类(在一个Sequential类中只能支持顺序连接的网络结构)，然后在Sequential实例中通过add函数添加网络层。keras把卷积层、池化层、RNN结构(LSTM、GRN)、全连接
层等常用的神经网络结构都做了封装，可以很方便地实现深层神经网络。在神经网络结构定义好之后，Sequential实例通过compile函数，指定优化函数、损失函数以及训练过程中需要监控
的指标等。Keras对优化函数、损失函数以及监控指标都有封装，同时也支持使用自定义的方式。最后在网络结构、损失函数和优化函数都定义好之后，Sequential实例通过fit函数来训练模型。
类似TFLearn中的fit函数，Keras的fit函数只须给出训练数据、batch大小和训练轮数，Keras就可以自动完成模型训练的整个过程。
'''

num_class = 10
img_rows, img_clos = 28, 28

#通过Keras封装好的API加载mnist数据。其中trainX就是一个60000 * 28 * 28的数组，trainY是每一张图片对应的数字
(trainX, trainY), (testX, testY) = mnist.load_data()

#因为不同的底层(Tensorflow或者MXNET)对输入的要求不一样，所以这里需要根据对图像编码的格式要求来设置输入层的格式
if k.image_data_format() == "channels_first":
    trainX = trainX.shape(trainX.shape[0], 1, img_rows, img_clos)
    testX = testX.shape(testX.shape[0], 1, img_rows, img_clos)
    #因为MNIST中的图片是黑白的，所以第一维的取值为1
    input_shape = (1, img_rows, img_clos)
else:
    trainX = trainX.shape(trainX.shape[0], img_rows, img_clos, 1)
    testX = testX.shape(testX.shape[0], img_rows, img_clos, 1)
    input_shape = (img_rows, img_clos, 1)

#将图像像素转化为0~1之间的实数
trainX = trainX.astype("float32")
testX = testX.astype("float32")
trainX /= 255.0
testX /= 255.0

#将标准答案转化为需要的格式(one-hot编码)
trainY = keras.utils.to_categorical(trainY, num_class)
testY = keras.utils.to_categorical(testY, num_class)

#使用Keras API定义模型
model = Sequential()
#一层深度为32，过滤器大小为5*5的卷积层
model.add(Conv2D(32, kernel_size=(5, 5), activation="relu", input_shape=input_shape, padding="same"))
#一层过滤器大小为2*2的最大池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
#一层深度为32，过滤器大小为5*5的卷积层
model.add(Conv2D(64, kernel_size=(5, 5), activation="relu", padding="same"))
#一层过滤器大小为2*2的最大池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
#将卷积层的输出拉直后作为下面全连接层的输入
model.add(Flatten())
#定义后面的全连接层
model.add(Dense(500, activation="relu"))
model.add(Dense(num_class, activation="softmax"))

#定义损失函数、优化函数和评测方法
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=["accuracy"])
#类似TFLearn中的训练过程，给出训练数据，batch大小、训练轮数和验证数据，keras可以自动完成模型训练的过程
model.fit(trainX, trainY, batch_size=128, epochs=20, validation_data=(testX, testY))
#在测试数据上计算准确率
score = model.evaluate(testX, testY)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
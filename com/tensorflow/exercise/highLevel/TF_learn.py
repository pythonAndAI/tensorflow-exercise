import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

#通过tflearn在Mnist数据集上实现LeNet5模型

#读取mnist数据集
trainX, trainY, testX, testY = mnist.load_data(data_dir="E:\\Alls\\软件\\tensorflow-mnist", one_hot=True)

#将图像数据reshape成卷积神经网络输入的格式
trainX = trainX.reshape([-1, 28, 28, 1])
print(trainX.shape)
testX = trainX.reshape([-1, 28, 28, 1])

#构建神经网络，这个过程和Tensorflow-Slim比较类似。input_data定义了一个placeholder来接入输入数据
net = input_data(shape=[None, 28, 28, 1], name="input")
#通过TFLearn封装好的API定义一个深度为32，过滤器为5*5，激活函数为RELU的卷积层
net = conv_2d(net, 32, 5, activation="relu")
#定义一个过滤器为2*2的最大池化层，步长默认和过滤器尺寸一样
net = max_pool_2d(net, 2)
#定义一个深度为64，过滤器为5*5，激活函数为RELU的卷积层
net = conv_2d(net, 32, 5, activation="relu")
net = max_pool_2d(net, 2)
#定义全连接层
net = fully_connected(net, 500, activation="relu")
net = fully_connected(net, 10, activation="softmax")

#使用TFLearn封装好的函数定义学习任务。指定优化器为sgd，学习率为0.01，损失函数为交叉熵
net = regression(net, optimizer="sgd", learning_rate=0.01, loss="categorical_crossentropy")
#通过定义的网络结构训练模型，并在指定的验证数据上验证模型的效果。TFLearn将模型的训练过程封装在一个类中，这个可以减少非常多的冗余代码
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=20, validation_set=([testX, testY]), show_metric=True)


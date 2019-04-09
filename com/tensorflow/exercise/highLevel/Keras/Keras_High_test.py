import keras
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model

#Keras以返回值的形式定义网络结构

'''
使用Keras_CNN.py中介绍的类似方法生成trainX、trainY、testX、testY，唯一的不同是这里只用了全连接层，所以不需要将输入整理成三维矩阵
'''

#定义输入，这里指定的维度不用考虑batch大小
inputs = Input(shape=(784,))
#定义一个全连接层，该层有500隐藏节点，使用Relu激活函数。这一层的输入为inputs。
x = Dense(500, activation="relu")(inputs)
#定义输出层。注意因为keras封装的categorical_crossentropy并没有将神经网络的输出再经过一层softmax，所以这里需要指定softmax作为激活函数
predictions = Dense(10, activation="softmax")(x)
#通过Model类创建模型，和Sequential类不同的是Model类在初始化的时候需要指定模型的输入和输出
model = Model(inputs=inputs, outputs=predictions)
#定义损失函数、优化函数和评测方法
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=["accuracy"])
#定义训练模型
model.fit('trainX', 'trainY', batch_size=128, epochs=20, validation_data=('testX', 'testY'))

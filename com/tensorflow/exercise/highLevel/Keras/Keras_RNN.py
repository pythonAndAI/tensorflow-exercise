from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

'''
keras除了能够很方便地处理图像问题，keras对于循环神经网络的支持也是非常出色的。有了keras API，循环神经网络的循环体结构也可以通过简单的一句命令完成。
以下代码给出了如何通过Keras实现自然语言情感分类问题。使用循环神经网络判断语言的情感（比如在以下例子中需要判断的一个评价是好评还是差评）和自然语言建模问题类似，
唯一的区别在于除了最后一个时间点的输出是有意义的，其他时间点的输出都可以忽略。情感分析模型结构参考model_structure.png
'''

#最多使用的单词数
max_features = 20000
#循环神经网络的截断长度
maxlen = 80
batch_size = 32

#加载数据并将单词转化为ID，max_features给出了最多使用的单词数。和自然语言模型类似，会将出现频率较低的单词替换为统一的ID，通过Keras封装的API会生成25000条训练数据和
#25000条测试数据，每一条数据可以被看成一段话，并且每段话都有一个好评或者差评的标签
(trainX, trainY), (testX, testY) = imdb.load_data(num_words=max_features)
print(len(trainX), "train sequences")
print(len(testX), "train sequences")

#在自然语言中，每一段话的长度是不一样的，但循环神经网络的循环长度是固定的，所以这里需要先将所有段落统一成固定长度。对于长度不够的段落，要使用默认值0来填充，对于超过长度的段落则直接忽略掉超过的部分
trainX = sequence.pad_sequences(trainX, maxlen=maxlen)
testX = sequence.pad_sequences(testX, maxlen=maxlen)
'''
输出统一长度之后的数据维度:
（"x_train shape:", (25000, 80)）
（"x_test shape:", (25000, 80)）
'''
print("trainX shape:", trainX.shape)
print("testX shape:", testX.shape)

#在完成数据预处理之后构建模型
model = Sequential()
#构建embedding层。128代表了embedding层的向量维度
model.add(Embedding(max_features, 128))
#构建LSTM层
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#构建最后的全连接层。注意在上面构建的LSTM层时只会得到最后一个节点的输出。如果需要输出每个时间点的结果，那么可以将return_sequences参数设置为True
model.add(Dense(1, activation="sigmoid"))

#定义损失函数、优化函数和评测指标。指定训练数据、训练轮数、batch大小以及验证数据
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(trainX, trainY, batch_size=batch_size, epochs=15, validation_data=(testX, testY))
#在测试数据上计算准确率
score = model.evaluate(testX, testY, batch_size=batch_size)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
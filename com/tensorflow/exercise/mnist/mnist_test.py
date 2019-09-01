import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#输入层节点，对于mnist数据集，这个数就等于图片的像素
INPUT_NODES = 784
#隐藏层的节点数
HIDDEN_NODES = 500
#输出层节点的数
OUTPUT_NODES = 10
#一个训练中batch中的训练数据个数。数字越大时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
BATCH_SIZE = 100
#基础学习率
LEARNING_RATE_BASE = 0.8
#学习率的衰减率
LEARNING_RATE_DECAY = 0.99
#描述模型复杂度的正则化项在损失函数中的系数
REGULARIZATION_RATE = 0.0001
#训练轮数
TRAINING_STEPS = 30000
#滑动平均衰减率
MAVING_AVERAGE_DECAY = 0.99

'''
指定神经网络的输入和所有参数，计算神经网络的前向传播结果，在这里定义了一个使用Rule激活函数的三层全连接神经网络。通过加入隐藏层实现了多层网络结构。
通过Rule激活函数实现了去线性化。在这个函数中也支持传入用于计算参数平均值的类，这样方便在测试时使用滑动平均模型
准确率达到98.31%
'''
def inference(input_temsor, avg_class, weigths1, basis1, weigth2, basis2):
    #当没有提供滑动平均类时，直接使用参数当前的取值。0
    if avg_class == None:
        #计算隐藏层的前向传播结果，这里直接使用rule激活函数
        layer1 = tf.nn.relu(tf.matmul(input_temsor, weigths1) + basis1)
        #计算输出层的前向传播结果，因为在计算损失函数时会一并计算softmax函数，所以这里不需要加入激活函数。
        return tf.matmul(layer1, weigth2) + basis2
    else:
        #首先使用avg_class.average函数来计算得出变量的滑动平均值，然后在计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_temsor, avg_class.average(weigths1)) + avg_class.average(basis1))
        return tf.matmul(layer1, avg_class.average(weigth2)) + avg_class.average(basis2)

#训练模型的过程
def train(mnist):
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODES], name='x-input')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_NODES], name='y-input')

    #生成隐藏层参数
    wegith1 = tf.Variable(tf.truncated_normal([INPUT_NODES, HIDDEN_NODES], stddev=0.1))
    basis1 = tf.Variable(tf.constant(0.1, shape=[HIDDEN_NODES]))
    #生成输出层参数
    wegith2 = tf.Variable(tf.truncated_normal([HIDDEN_NODES, OUTPUT_NODES], stddev=0.1))
    basis2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODES]))

    #计算在当前参数下神经网络前向传播的结果，这里给出的用于计算滑动平均类为None，所以函数不会使用滑动平均类
    y = inference(x, None, wegith1, basis1, wegith2, basis2)

    #定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量(trainable=False)。
    #在使用Tensorflow训练模型时，一般会把代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。给定训练轮数变量可以加快训练早期变量的更新速度。
    variable_averages = tf.train.ExponentialMovingAverage(MAVING_AVERAGE_DECAY, global_step)

    #在所有代表神经网络参数的变量上使用滑动平均，其他辅助变量(比如global_step)就不需要了。
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #计算使用滑动平均之后的前向传播结果，前面结果过滑动平均不会改变变量本省的取值，而是会维护一个影子变量来记录其滑动平均值，
    #所以当需要使用这个滑动平均值时，需要明确调用average函数
    average_y = inference(x, variable_averages, wegith1, basis1, wegith2, basis2)

    #定义损失函数，此分类问题只有一个正确结果，所以可以使用sparse_softmax_cross_entropy_with_logits来加速交叉熵的计算。
    #第一个参数为前向传播结果，第二个参数为正确答案。因为标准答案是一个长度为10的一维数组，而该函数需要提供一个正确答案的数字，
    #所以需要使用tf.argmax函数来得到正确答案对应的类别编号。y-->[100, 10], y_--->[100, 10], tf.argmax(y_, 1)--->[1, 100]
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    #计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularization = regularizer(wegith1) + regularizer(wegith2)
    #总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    #设置指数衰减的学习率,第一个参数为初始学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减。第二个参数为当前迭代的轮数。
    #第三个参数为过完所有的训练数据需要的迭代次数，第四个参数为学习率衰减速度
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    #定义优化器。需要增加global_step=global_step，优化器每次给global_step加1，用于改变学习率的滑动平均影子变量的值
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值，为了一次完成多个操作，
    #Tensorflow提供了tf.control_dependencies和tf.group两种机制，下面两行程序和
    #train_op = tf.group(train_step, variable_averages_op)是相等的
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    #检验使用了滑动平均模型的神经网络前向传播结果是否正确。tf.argmax(average_y, 1)计算每一个样例的预测答案。其中average_y是一个batch_size*10
    #的二维数组，每一行表示一个样例的前向传播结果，tf.argmax的第二个参数"1"表示选取最大值的操作仅在第一个维度中进行，也就是说，只在每一行选取
    #最大值对应的下标，于是得到的结果是一个长度为batch的一维数组，这个一维数组中的值就表示了每一个样例对应的数字识别结果。tf.equal判断两个张量
    #的每一维是否相等，如果相等返回True，否则返回False。
    #其中tf.argmax(average_y,1)获取每一维数字中的最大值的下标，返回一个全是下标的一维数组。tf.equal对比两个张量每一维数字是否相等。相等返回True，不等返回False.
    #所以correct_prediction是一个里面全是True或False的一维数组。correct_prediction和accuracy两步中用到的函数详情可参考test.py的练习
    #bb = tf.constant([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]], dtype=tf.float32, name="b")
    #print(sess.run(tf.argmax(bb, 1)))
    #[0 3 1]
    correct_prediction = tf.equal(tf.argmax(average_y,1), tf.argmax(y_, 1))

    #这个运算首先将一个布尔型的数值转换为实数型，然后计算平均值，这个平均值就是模型在这一组数据上的正确率
    #其中tf.cast把correct_prediction的一维数组中的True换为1，False换为0。reduce_mean计算平均值。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        #准备测试数据，在真实的应用中，这部分数据在训练时是不可见的，这个数据只是作为模型优劣的最后评价标准
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        #迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
            #计算滑动平均模型在验证数据上的结果。因为MNIST数据集比较小，所以一次可以处理所有的验证数据。为了计算方便，本样例程序没有
            #将验证数据划分为更小的batch。当神经网络模型比较复杂或者验证数据比较大时，太大的batch会导致计算时间过长甚至发生内存溢出的错误
            #validation数据因为没有使用batch，所以一次会检测所有数据，前向传播结果为一个5000*10的矩阵
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy " "using average model is %g " % (i, validate_acc))
                # print("global_step is==>", sess.run(global_step))
                # print("validation y", sess.run(y, feed_dict={x: mnist.validation.images}))
                # print("validation average_y", sess.run(average_y, feed_dict={x: mnist.validation.images}))
                # print("validation size", len(sess.run(average_y, feed_dict={x: mnist.validation.images})), len(sess.run(average_y, feed_dict={x: mnist.validation.images})[0]))
            #产生这一轮使用的一个batch的训练数据，并进行训练过程。前向传播结果为一个100*10的矩阵
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
            # print("weigths is===>", sess.run(wegith1[0][0:10]))
            # print(sess.run(variable_averages_op))
            # print("weigths2 is===>", sess.run(variable_averages.average(wegith1)[0][0:10]))
            # print("xs is==>", xs[0],"ys is==>", ys[0])
            # print("learning_rate is==>", sess.run(learning_rate))
            #print("global_step is==>", sess.run(global_step))
            # print("train y", sess.run(y, feed_dict={x: xs}))
            # print("train average_y", sess.run(average_y, feed_dict={x: xs}))
            # print("train size", len(sess.run(average_y, feed_dict={x: xs})),
            #       len(sess.run(average_y, feed_dict={x: xs})[0]))

        #在训练结束之后，在测试数据上检测神经网络模型的最终正确率
        #前向传播结果为一个10000*10的矩阵
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        # print("test y", sess.run(y, feed_dict={x: mnist.test.images}))
        # print("test average_y", sess.run(average_y, feed_dict={x: mnist.test.images}))
        # print("test size", len(sess.run(average_y, feed_dict={x: mnist.test.images})),
        #       len(sess.run(average_y, feed_dict={x: mnist.test.images})[0]))
        print("After %d test step(s), test accuracy " "using average model is %g " % (TRAINING_STEPS, test_acc))

#主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("E:\\Alls\\软件\\tensorflow-mnist", one_hot=True)
    train(mnist)

#Tensorflow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == "__main__":
    tf.app.run()


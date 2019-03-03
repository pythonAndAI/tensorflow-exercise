import tensorflow as tf

#variable的作用就是保存和更新神经网络中的参数参数指定默认值，没指定的情况下为0
def variable_produce():
    #1.用随机数的方式产生初始值。会生成一个2*3的矩阵。矩阵中的元素的均值为0，标准差为2的随机数。可通过mean----->shape参数必须是个数组格式[]
    weights = tf.Variable(tf.random_normal([5, 3]))
    weights2 = tf.Variable(tf.random_normal([3, 1]))
    #2.用常量的方式产生初始值
    biases = tf.Variable(tf.zeros([2]) + 0.1)
    #3.通过其他变量的初始值初始化新的变量.得到的初始化值和weights的初始化值相同
    weights_new = tf.Variable(weights.initialized_value())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # sess.run(weights.initializer)
        print(sess.run(weights))
        print(sess.run(weights2))
        print(sess.run(biases))
        print(sess.run(weights_new))
        #矩阵相乘。相乘的两个矩阵必须是第一个矩阵的列等于第二个矩阵的行。相乘得到的新的矩阵的大小为---->第一个矩阵的行*第二个矩阵的列
        print(sess.run(tf.matmul(weights, weights2)))

#以下为Tensorflow目前支持的所有随机数生成器
def random_type():
    #正态分布，主要参数-->平均值、标准值、取值类型
    random_normal = tf.Variable(tf.random_normal([2, 3]))
    #正态分布(如果随机出来的值偏离平均值超过2个标准差，那么这个值就会被重新随机)，主要参数-->平均值、标准值、取值类型
    truncated_normal = tf.Variable(tf.truncated_normal([2, 3]))
    #均匀分布，主要参数-->最小，最大取值、取值类型
    random_uniform = tf.Variable(tf.random_uniform([2, 3]))
    #Gamma分布,主要参数-->形状参数alpha、尺度参数beta、取值类型
    random_gamma = tf.Variable(tf.random_gamma([2, 3], 1))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(random_normal))
        print(sess.run(truncated_normal))
        print(sess.run(random_uniform))
        print(sess.run(random_gamma))

#以下为Tensorflow常用的常数定义方法
def constant_type():
    #产生全0的数组
    constant_zero = tf.Variable(tf.zeros([2, 3]))
    constant_zero2 = tf.Variable(tf.zeros(3))
    #产生全1的数组
    ones_constant = tf.Variable(tf.ones([2, 3]))
    #产生一个全部为给定数字的数组
    fill_constant = tf.Variable(tf.fill([2, 3], 5.3))
    #产生一个给定值的常量
    constant = tf.Variable(tf.constant(1, dtype=tf.float32))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(constant_zero))
        print(sess.run(constant_zero2))
        print(sess.run(ones_constant))
        print(sess.run(fill_constant))
        print(sess.run(constant))

'''
1.所有的变量都会被自动的加入到GraphKeys.VARIABLES这个集合中，可以通过tf.global_variables()函数拿到当前计算图上所有的变量
2.当构建机器学习模型时，比如神经网路，可以通过变量声明函数中的trainable参数来区分需要优化的参数。
如果声明变量时参数trainable为True(默认为True)，那么这个变量将会被加入到GraphKeys.TRAINABLE_VARIABLES集合。
在Tensorflow中可以通过tf.trainable_variables()函数得到所有需要优化的参数。Tensorflow中提供的神经网络优化算法
会将GraphKeys.TRAINABLE_VARIABLES集合中的变量作为默认的优化对象
'''
def trainable_variable():
    w1 = tf.Variable(tf.random_normal([2, 3]), name="w1")
    w2 = tf.Variable(tf.random_normal([3, 1]), name="w2")
    w3 = tf.Variable(tf.random_normal([5, 2]), name="w3", trainable=False)
    w4 = tf.Variable(tf.random_normal([1, 4]), name="w4")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #可以通过tf.trainable_variables()函数得到所有需要优化的参数
        print(sess.run(tf.trainable_variables()))
        print("============================")
        #可以通过tf.global_variables()函数拿到当前计算图上所有的变量
        print(sess.run(tf.global_variables()))


if __name__ == "__main__":
    # variable_produce()
    # random_type()
    # constant_type()
    trainable_variable()
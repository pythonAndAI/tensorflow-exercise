import tensorflow as tf

'''
Tensorflow提供了placeholder机制用于提供输入数据。placeholder相当于定义了一个位置，这个位置中的数据在程序运行时在指定，这样在程序中就不需要
生成大量常量来提供输入数据，而只需要将数据通过placeholder传入Tensorflow计算图
'''
def place_holder(batch=True):
    w1 = tf.Variable(tf.random_normal([3, 2]))
    w2 = tf.Variable(tf.random_normal([2, 1]))

    if batch:
        x = tf.placeholder(dtype=tf.float32, shape=(1, 3), name="x")
    else:
        x = tf.placeholder(dtype=tf.float32, shape=(3, 3), name="x")
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    #feed_dict是一个字典
    if batch:
        feed_dict = {x: [[0.7, 0.8, 0.9]]}
    else:
        #在训练神经网络时需要每次提供一个batch的训练样例。如下，如果将输入1*2矩阵改成n*2的矩阵，那么就可以得到n个样例的前行传播结果
        #其中n*2的矩阵的每一行为一个样例数据，这样前行传播的结果为n*1的矩阵，这样矩阵的每一行就代表一个样例的前向传播结果
        feed_dict = {x: [[0.7, 0.8, 0.9], [0.2, 0.3, 0.4], [0.1, 0.5, 0.6]]}
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    result = sess.run(y, feed_dict=feed_dict)
    sess.close()
    print(result)
    return result

#定义损失函数
def loss_func():
    sess = tf.Session()
    y_result = place_holder(batch=False)
    y = tf.sigmoid(y_result)

    print(sess.run(y))
    y_ = tf.constant([[0.1], [0.3], [0.6]])
    cross_entropy = - tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y)* tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
    #优化器,共支持10种优化器
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    sess.close()
    pass

if __name__ == "__main__":
    place_holder()


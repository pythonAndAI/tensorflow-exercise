import tensorflow as tf
import  numpy as np
#一个案例
'''
1.定义神经网络的结构和前向传播的输出结果
2.定义损失函数以及选择反向传播优化的算法
3.生成会话（tf.Session）并且在训练数据上反复运行反向传播优化算法
'''
def case():
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    batch_size = 8
    x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="x_input")
    y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y_input")

    #定义神经网络的前行传播
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)
    #定义损失函数和反向传播
    y = tf.sigmoid(y)
    cross_entropy = - tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    #通过随机数生成一个模拟数据集
    dataset_size = 128
    X = np.random.rand(dataset_size, 2)
    Y = [[int (x1+x2 < 1)] for (x1, x2) in X]

    with tf.Session() as sess:
        #初始化变量
        init = tf.global_variables_initializer()
        sess.run(init)

        print("tarin before w1 is:", sess.run(w1))
        print("tarin before w2 is:", sess.run(w2))

        #设定训练的次数
        STEPS= 5000
        for i in range(STEPS):
            start = ( i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            #start:end以8为单位截取X---->[0:8],[8:16]等
            #通过选取的样本训练神经网络并更新参数
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
            if i % 1000 ==0:
                #每隔一段时间计算在所有数据上的交叉熵并输出
                print("first", i, "cross_entropy is:", sess.run(cross_entropy, feed_dict={x: X, y_: Y}))
        print("tarin after w1 is:", sess.run(w1))
        print("tarin after w2 is:", sess.run(w2))

    pass

if __name__ == "__main__":
    case()
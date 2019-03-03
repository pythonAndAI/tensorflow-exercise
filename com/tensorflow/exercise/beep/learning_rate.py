import tensorflow as tf

def learning_rate():
    #指数衰减法设置学习率。可以先设置一个较大的学习率来快速得到一个比较优的解，然后随着迭代的继续逐步减少学习率
    #tf.train.exponential_decay函数实现了如下:decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    #其中decayed_learning_rate为每一轮优化时使用的学习率，learning_rate为事先设定的初始学习率，decay_rate为衰减系数，decay_steps为衰减速度，
    # staircase参数可以设置不同的衰减方式，当staircase为true时，global_step / decay_steps会被转化为整数，这就使学习率成为一个阶梯函数
    # tf.train.exponential_decay(learning_rate,
    #                   global_step,
    #                   decay_steps,
    #                   decay_rate,
    #                   staircase=False,)
    pass

def learning_rate_exercise():
    global_step = tf.Variable(0)
    w1 = tf.Variable(tf.random_normal([2, 3]))
    w2 = tf.Variable(tf.random_normal([3, 1]))
    y_ = tf.constant([1.0])

    a = tf.placeholder(dtype=tf.float32, shape=(1, 2))

    x = tf.matmul(a, w1)
    x = tf.sigmoid(x)
    y = tf.matmul(x, w2)
    y = tf.sigmoid(y)
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1-y) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
    learning_rate = tf.train.exponential_decay(0.3, global_step, 200, 0.96, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        STEPS = 5000
        for i in range(STEPS):
            sess.run(train_step, feed_dict={a: [[0.12, 0.13]]})
            if i % 500 == 0:
                print("learning_rate is:", sess.run(learning_rate))
                print("cross_entropy is:", sess.run(cross_entropy, feed_dict={a: [[0.12, 0.13]]}))
                print("global_step is:", sess.run(global_step))


if __name__ == "__main__":
    # learning_rate()
    learning_rate_exercise()
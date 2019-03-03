import tensorflow as tf

#过拟合是指当一个模型过为复杂之后，它可以很好地“记忆”每一个训练数据中随机噪音的部分而忘记了要去“学习”训练数据中通用的趋势。
#过拟合完全可以记住所有训练数据的结果而使损失函数为0，但是并不能很好的对位置数据做出判断。因为它过度拟合了训练数据中的噪音而忽略了问题的整体规律
#为了避免过拟合问题，一个非常常用的方法是正则化。正则化的思想就是在损失函数中加入刻画模型复杂程度的指标.有l1和l2正则化
def regularization():
    #如下：loss为定义的损失函数，他由两部分组成，一部分是均方误差损失函数，它刻画了模型在训练数据上的表现。另一部分就是正则化，它防止模型过度模拟训练数据中的随机噪音
    #其中lambda参数表示了正则化项的权重，w为需要计算正则化损失的参数
    # loss = tf.reduce_mean(tf.square(y_ - y) + tf.contrib.layers.l1_regularizer(lambda)(w))
    weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
    with tf.Session() as sess:
        print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weights)))
        print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights)))
    pass

def get_weight(shape, labda):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(labda)(var))
    return var

def overfitting_exercise():
    x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
    y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))
    layers = [3, 1]
    layers2 = [2, 3]
    cur_layer = x
    for i in range(2):
        weight = get_weight([layers2[i], layers[i]], 0.001)
        bias = tf.Variable(tf.constant(0.1, shape=[layers[i]]), dtype=tf.float32)
        cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)

    # cross_entyopy = - tf.reduce_mean(y_ * tf.log(tf.clip_by_value(cur_layer, 1e-10, 1.0)) + (1-cur_layer) * tf.log(tf.clip_by_value(1-cur_layer, 1e-10, 1.0)))
    #最后只有一个输出，所以用到均方误差
    cross_entyopy = tf.reduce_mean(tf.square(y_ - cur_layer))
    tf.add_to_collection("losses", cross_entyopy)
    loss = tf.add_n(tf.get_collection("losses"))
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    STEPS = 5000
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(STEPS):
            sess.run(train_step, feed_dict={x: [[0.234, 0.569]], y_: [[0.99]]})
            if i % 500 == 0:
                print("loss is:", sess.run(loss, feed_dict={x: [[0.234, 0.569]], y_: [[0.99]]}))

if __name__ == "__main__":
    # regularization()
    overfitting_exercise()
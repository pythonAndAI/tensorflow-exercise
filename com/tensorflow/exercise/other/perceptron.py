import tensorflow as tf
import numpy as np
'''
感知机实现异或问题
'''

if __name__ == "__main__":
    x_data = np.array([[1 ,0] ,[1 ,1], [0 ,1], [0 ,0]], dtype=np.float32)
    y_data = np.array([[1], [0] , [1], [0]], dtype=np.float32)
    # weigths = tf.Variable(tf.random_uniform([2, 4], -1 ,1), name="weigths")
    #
    # weigths2 = tf.Variable(tf.random_uniform([4, 1], -1 ,1), name="weigths2")

    weigths = tf.Variable(tf.truncated_normal([2, 4], stddev=1), name="weigths")

    weigths2 = tf.Variable(tf.truncated_normal([4, 1], stddev=1), name="weigths2")
    b1 = tf.Variable(tf.zeros(4, dtype=np.float32), name='b1')
    b2 = tf.Variable(tf.zeros(1, dtype=np.float32), name='b2')

    x = tf.placeholder(shape=[4, 2], dtype=tf.float32, name="input-data")
    y_ = tf.placeholder(shape=[4, 1], dtype=tf.float32, name="y_-data")
    h1 = tf.matmul(x, weigths) + b1
    #经过sigmoid函数得到输出
    h = tf.sigmoid(h1)
    y = tf.sigmoid(tf.matmul(h, weigths2) + b2)

    print(y.shape, y_.shape, tf.argmax(y_, 1).shape)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    loss = tf.nn.l2_loss(y - y_)
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        h1_run = sess.run(h1, feed_dict={x: x_data})
        h_run = sess.run(h, feed_dict={x: x_data})
        print(h1_run)
        print(h_run)
        #
        # for i in range(100000):
        #     if i % 10000 == 0:
        #         loss_value = sess.run(loss, feed_dict={x: x_data, y_ : y_data })
        #         print("第", i, "次为:",  loss_value)
        #     sess.run(train_op, feed_dict={x: x_data, y_ : y_data })





import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Keras和原生态Tensorflow API联合起来解决MNIST问题
#通过和原生态Tensorflow更紧密地结合，可以使建模的灵活性进一步提高，但是同时也会损失一部分封装带来的易用性。所以在实际问题中可以根据需求合理地选择封装的程度

mnist = input_data.read_data_sets("E:\\Alls\\软件\\tensorflow-mnist", one_hot=True)

#通过Tensorflow中的placeholder定义输入。类似地，Keras封装的网络层结构也可以支持使用前面介绍的输入队列。这样可以有效避免一次性加载所有数据的问题
X = tf.placeholder(tf.float32, shape=[None, 784], name="input")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="label")

#直接使用Tensorflow中提供的Keras API定义网络层结构
net = tf.keras.layers.Dense(500, activation="relu")(X)
y = tf.keras.layers.Dense(10, activation="softmax")(net)

#定义损失函数和优化方法。注意这里可以混用Keras的API和原生态Tensorflow的API
loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_, y))
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#定义预测的正确率作为指标
acc_value = tf.reduce_mean(tf.keras.metrics.categorical_crossentropy(y_, y))

#使用原生态Tensorflow的方式训练模型。这样可以有效地实现分布式
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(10000):
        xs, ys = mnist.train.next_batch(100)
        _, loss_value = sess.run([train_op, loss], feed_dict={X:xs, y_:ys})
        if i % 1000 == 0:
            print("After %d training steps, loss on training batch is %g" % (i, loss_value))

    print(acc_value.eval(feed_dict={X:mnist.test.images, y_:mnist.test.labels}))
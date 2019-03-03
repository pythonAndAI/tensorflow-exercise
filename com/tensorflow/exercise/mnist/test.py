import tensorflow as tf

if __name__ == "__main__":
    w1 = tf.Variable(tf.random_normal([2, 3]))
    w2 = tf.Variable(tf.random_normal([2, 3]))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(w1))
        print(sess.run(w2))
        print(sess.run(tf.argmax(w1, 1)))
        print(sess.run(tf.argmax(w2, 1)))
        print(sess.run(tf.equal(tf.argmax(w1, 1), tf.argmax(w2, 1))))
        print(sess.run(tf.cast(tf.equal(tf.argmax(w1, 1), tf.argmax(w2, 1)), tf.float32)))
        print(sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(w1, 1), tf.argmax(w2, 1)), tf.float32))))
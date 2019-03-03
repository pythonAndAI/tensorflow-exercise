import tensorflow as tf

#滑动平均模型，在采用随机梯度下降算法训练神经网络时，使用滑动平均模型在很多应用中都可以在一定程度提高最终模型在测试数据上的表现
#使用tf.train.ExponentialMovingAverage()函数来实现滑动平均模型，在初始化时需要提供一个衰减率(decay)，这个衰减率将用于控制模型更新的速度
#ExponentialMovingAverage对每一个变量会维护一个影子变量(shadow variable)，这个影子变量的初始值就是相应变量的初始值，而每次运行变量更新时，
# 影子变量的值会更新为:shadow_variable = decay * shadow_variable + (1 - decay) * variable，其中shadow_variable为影子变量，variable为待更新的值
#decay为衰减率。从公式可以看出，decay决定了模型更新的速度，decay越大模型越趋于稳定。所以在实际应用中decay一般会设置为非常接近1的值(0.99或0.999)
#为了使模型在训练前期可以更新的更快，ExponentialMovingAverage还提供了num_updates参数来动态设置decay的大小。则每次的衰减率为min{decay, (1+num_updates)/(10+num_updates)}

if __name__ == "__main__":
    #定义一个变量用于计算滑动平均，这个变量的初始值为0,。
    v1 = tf.Variable(0, dtype=tf.float32)
    #这里step变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率
    step = tf.Variable(0, trainable=False)
    #定义一个滑动平均的类，初始化时给定了衰减率(0.99)和控制衰减率的变量step
    ema = tf.train.ExponentialMovingAverage(0.99, step)

    #定义一个更新变量滑动平均的操作，这里需要给定一个列表，每次执行这个操作时，这个列表中的变量都会被更新
    maintain_averages_op = ema.apply([v1])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #通过ema.average(v1)获取滑动平均之后变量的取值，在初始化之后变量v1的值和v1的滑动平均都为0
        print(sess.run([v1, ema.average(v1)]))

        #更新变量v1的值为5
        sess.run(tf.assign(v1, 5))

        #更新v1的滑动平均值，衰减率计算公式为min{decay, (1+num_updates)/(10+num_updates)}=min{0.99, (1+step)/(10+step)=0.1}=0.1
        #v1的滑动平均会被更新为hadow_variable = decay * shadow_variable + (1 - decay) * variable=0.1 * 0 + 0.9 * 5 = 4.5
        #下面的计算同理
        # sess.run(maintain_averages_op)
        print(sess.run([v1, ema.average(v1)]))

        # 更新变量step的值为10000
        sess.run(tf.assign(step, 10000))
        # 更新变量v1的值为10
        sess.run(tf.assign(v1, 10))

        # v1的滑动平均会被更新为0.99 * 4.5 + 0.01 * 10 = 4.555
        # sess.run(maintain_averages_op)
        print(sess.run([v1, ema.average(v1)]))

        #再次更新滑动平均值，得到更新后的值为0.99*4.555+0.01*10=4.60945
        # sess.run(maintain_averages_op)
        print(sess.run([v1, ema.average(v1)]))
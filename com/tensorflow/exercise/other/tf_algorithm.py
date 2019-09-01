import tensorflow as tf
import cv2

'''
交叉熵，计算两个概率分布之间的距离。y=-sum p(x)log q(x) --->p为真实数据，q为输出数据
sparse_softmax_cross_entropy_with_logits-->等于先对logits预测值计算softmax，得到预测值的概率分布。在计算交叉熵 y = sum(p(x) log q(x))
'''
def cross_entropy():
    param1 = tf.constant([[0.01, 0.015, 0.14], [0.23, 0.21, 0.19], [0.175, 0.182, -0.02], [0.136, 0.02, 0.024]], dtype=tf.float32,
                         name="param1")
    param2 = tf.constant([[0.032, 0.26, -0.366], [ 0.012, 0.0656, 0.05], [0.035, 0.027, 0.29]], dtype=tf.float32, name="param2")

    y_ = tf.constant([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=tf.float32, name="y_")

    # 4 * 3 -->[[ 0.0054     0.007364   0.03769  ]
    #  [ 0.01653    0.078706  -0.01858  ]
    #  [ 0.007084   0.0568992 -0.06075  ]
    #  [ 0.005432   0.03732   -0.041816 ]]
    matmul_value = tf.matmul(param1, param2)

    # 4 * 3 ---> np.exp(0.0054) / ( np.exp(0.0054) + np.exp(0.007364) + np.exp(0.03769 )) ...
    softmax_value = tf.nn.softmax(matmul_value)
    #[1.0778499 1.1084467 1.0439495 1.0940193]  ---> sum等于4.3242655
    #- （0.0 * log(0.32951286) + 0.0 * log(0.33016068) + 1.0 * log(0.3403265)） = 1.0778499
    cross_entropy1 = -tf.reduce_sum(y_ * tf.log(softmax_value))
    #计算得到cross_entropy1=cross_entropy2=cross_entropy3
    cross_entropy2 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=matmul_value, labels=tf.arg_max(y_, 1)))
    cross_entropy3 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=matmul_value, labels=y_))


    with tf.Session() as sess:
        print("matmul_value is", sess.run(matmul_value))
        print("softmax_value is", sess.run(softmax_value))
        print("cross_entopy1 is", sess.run(cross_entropy1))
        print("cross_entropy2 is", sess.run(cross_entropy2))
        print("cross_entropy3 is", sess.run(cross_entropy3))

def activation_sigmoid():
    param1 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, -8.0, 9.0], [11.0, 2.0, -4.0]], dtype=tf.float32,
                         name="param1")
    #1 / 1 + np.exp(-1.0), 1 / 1 + np.exp(-2.0) ...
    sigmoid_value = tf.nn.sigmoid(param1)
    #R（x）= max（0，x），即如果x <0，R（x）= 0，如果x> = 0，则R（x）= x
    relu_value = tf.nn.relu(param1)
    with tf.Session() as sess:
        print("sigmoid_value is", sess.run(sigmoid_value))
        print("relu_value is", sess.run(relu_value))
'''
过拟合
1.regularizer方式
1.1 l1--> λR(w) = λΣ|wi| --->regularizer_param1
1.2 l2--> λR(w) = λ/2 Σ|wi2|(平方) --->regularizer_param2
2.dropout
dropout_param
'''
def overfitting_test():
    param1 = tf.constant([[0.01, 0.015, 0.14], [0.23, 0.21, 0.19]], dtype=tf.float32, name="param1")
    param2 = tf.constant([[0.01, 0.015, 0.14], [0.23, 0.21, 0.19], [0.175, 0.182, 0.021], [0.136, 0.022, 0.024]],
                         dtype=tf.float32, name="param1")
    #0.0079499995 = 0.01 * (0.01 + 0.015 + 0.14 + 0.23 + 0.21 + 0.19)
    regularizer_param1 = tf.contrib.layers.l1_regularizer(0.01)(param1)
    #0.000765125 = 0.01 / 2 * (0.01 * 0.01 + 0.015 * 0.015 + 0.14 * 0.14 + 0.23 * 0.23 + 0.21 * 0.21 + 0.19 * 0.19)
    regularizer_param2 = tf.contrib.layers.l2_regularizer(0.01)(param1)
    #以一定概率让激活值变为0
    dropout_param = tf.nn.dropout(param2, 0.5)
    with tf.Session() as sess:
        print("regularizer_param is", sess.run(regularizer_param1))
        print("regularizer_param2 is", sess.run(regularizer_param2))
        print("dropout_param is", sess.run(dropout_param))

def lrn_test():
    param = tf.Variable(tf.random_uniform([1, 1, 1, 5]))
    lrn_param = tf.nn.lrn(param, 2, 1, 1, 1)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(param))
        print(sess.run(lrn_param))
    pass

def bn_test():
    param = tf.Variable(tf.random_uniform([2, 2, 2, 2]))
    fc_mean, fc_var = tf.nn.moments(param, axes=[0])
    scale = tf.Variable(tf.ones([2]))
    shift = tf.Variable(tf.zeros([2]))
    epsilon = 0.001
    bn_param = tf.nn.batch_normalization(param, fc_mean, fc_var, shift, scale, epsilon)

    bn_param2 = (param - fc_mean) / tf.sqrt(fc_var + epsilon)
    bn_param2 = bn_param2 * scale + shift

    bn_param3 = tf.layers.batch_normalization(param, axis=0, epsilon=epsilon)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("param is", sess.run(param))
        print("bn_param is", sess.run(bn_param))
        print("fc_mean is", sess.run(fc_mean))
        print("fc_var is", sess.run(fc_var))
        print("scale is", sess.run(scale))
        print("shift is", sess.run(shift))
        print("bn_param2 is", sess.run(bn_param2))
        print("bn_param3 is", sess.run(bn_param3))

if __name__ == "__main__":
    #activation_sigmoid()
    #cross_entropy()
    #overfitting_test()
    # lrn_test()
    # bn_test()
    cv2.imread()
import tensorflow as tf

'''
1.神经网络模型的效果以及优化的目标是通过损失函数来定义的
2.分类问题和回归问题是监督学习的两大类别
3.交叉熵刻画的是两个概率分布之间的距离
4.H(p,q)=-(后面计算的加权和)p(x)log q(x)，其中p为正确答案，q为预测值
5.Softmax回归在Tensorflow中，它只是一个额外的处理层，将神经网络的输出变成一个概率分布。
6.从交叉熵的公式中可以看到交叉熵函数不是对称的（H(p,q) != H(q,p)）,它刻画的是通过概率分布q来表达概率分布p的困难程度。
因为正确答案是希望得到的结果，所以当交叉熵作为神经网络的损失函数时，p为正确答案，q为预测值。交叉熵刻画的是两个概率分布的
距离，也就是说交叉熵值越小，两个概率分布越接近
'''
def cross_entyopy():
    #分类问题的交叉熵定义
    #完整的交叉熵定义为
    # cross_entyopy = - tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 0.1)))
    #下面拆分每一步
    v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v2 = tf.constant([1.0, 2.0, 3.0])
    v3 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    v4 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    with tf.Session() as sess:
        #tf.clip_by_value函数控制一个张量的值在一定范围之内，如下，控制v的值在2.5-4.5之间
        print(sess.run(tf.clip_by_value(v, 2.5, 4.5)))
        #对张量中的值依次求对数
        print(sess.run(tf.log(v2)))
        #矩阵*操作相乘，不等于矩阵相乘(tf.matmul)。是每个位置上对应元素的乘积
        print(sess.run(v3 * v4))
        #矩阵相乘
        print(sess.run(tf.matmul(v3, v4)))
        #对整个矩阵做平均。等于1.0+2.0+3.0+4.0+5.0+6.0=21，除于个数6
        print(sess.run(tf.reduce_mean(v))== 21/6)
    pass

def softmax_entropy():
    #整合交叉熵和Softmax,其中y是神经网络的输出，y_是标准输出
    # tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    #在只有一个正确答案的分类问题上，Tensorflow提供了如下函数来进一步加速计算过程。0
    # tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)
    pass

#回归问题解决的是对具体数值的预测。比如房价预测等。这些问题需要预测的不是一个事先定义好的类别，而是一个任意实数。
#解决回归问题的神经网络一般只有一个输出节点，这个节点的输出值就是预测值。这种问题，最常用的损失函数是均方误差(MSE)
def squared_error():
    #其中y代表了神经网络的输出答案，y_代表了标准答案
    # mse = tf.reduce_mean(tf.square(y_ - y))
    pass

if __name__ == "__main__":
    cross_entyopy()
import tensorflow as tf
'''
会话拥有并管理Tensorflow程序运行时的所有资源。所有计算完成之后需要关闭会话来帮助系统回收资源，否则就可能出现资源泄露的问题
'''
def direct_sess():
    a = tf.constant(1, tf.float32, name="a")
    sess = tf.Session()
    print(sess.run(a))
    #需要关闭资源
    sess.close()

def with_sess():
    a = tf.constant(1, tf.float32, name="a")
    #创建一个会话，并通过python中的上下文管理器来管理这个会话
    with tf.Session() as sess:
        print(sess.run(a))
    #不需要在调用session.close()函数来关闭会话
    #当上下文退出时会话关闭和资源释放也自动完成了

#创建默认会话，同Tensorflow的默认计算图。不过Tensorflow不会自动生成默认会话，需要手动指定
def default_sess():
    a = tf.constant(1, tf.float32, name="a")
    sess = tf.Session()
    with sess.as_default():
        #获取张量的取值
        print(a.eval())

    sess2 = tf.Session()
    #以下两个命令有相同的功能
    print(sess2.run(a))
    print(a.eval(session=sess2))
    sess2.close()

    #在交互式环境下可以通过InteractiveSession()函数直接构建默认会话。可以省去将产生的会话注册为默认会话的过程。
    sess3 = tf.InteractiveSession()
    print(a.eval())
    sess3.close()

    '''
    无论用那种方式产生会话，都可以通过ConfigProto Protocol Buffer来配置需要生成的会话
    通过ConfigProto可以配置类似并行的线程数、GPU分配策略、运行超时时间等参数。在这些参数中，最常用的有两个。分别是：
    1.allow_soft_placement:这是一个布尔值，当为True时，在以下任意一个条件成立时，GPU上的运算可以放到CPU上进行；
    1.>运算无法再GPU上运行
    2.>没有GPU资源(比如运算被指定在第二个GPU，但是机器只有一个GPU,可以通过tf.Graph().device('/gpu:0')这种方式指定计算运行的设备)
    3.>运算输出包含对CPU计算结果的引用
    2.log_device_placement，当它为True时日志中将会记录每个节点被安排在哪个设备上以方便调试。而在生产环境中将这个参数设置为False可以减少日志量。
    '''
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess4 = tf.InteractiveSession(config=config)
    print(sess4.run(a))
    sess4.close()

if __name__ == '__main__':
    # direct_sess()
    # with_sess()
    default_sess()
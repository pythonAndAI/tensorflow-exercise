import tensorflow as tf

'''
当神经网络的结构更加复杂、参数更多时，就需要一个更好的方式来传递和管理神经网络中的参数。Tensorflow提供了通过变量名称来创建或者获取一个变量的机制。
通过这个机制，在不同的函数中可以直接通过变量的名字来使用变量，而不需要将变量通过参数的形式到处传递。这个机制主要是通过tf.get_variable和tf.variable_scope
函数实现的。而且tf.get_variable和tf.Variable创建变量的功能是等价的。
其中tf.get_variable的initializer函数有如下：
tf.ones_initializer()                  将变量设置为全1                                         主要参数为变量维度
tf.zeros_initializer()                 将变量设置为全0                                         主要参数为变量维度
tf.constant_initializer()              将变量初始化为给定常量                                  主要参数为常量的取值
tf.random_normal_initializer()         将变量初始化为满足正态分布的随机值                      主要参数为正态分布的均值和标准差
tf.random_uniform_initializer()        将变量初始化为满足平均分布的随机值                      主要参数为最大值、最小值
tf.uniform_unit_scaling_initializer()  将变量初始化为满足平均分布但不影响输出数量级的随机值    主要参数为factor(产生随机值时乘以的系数)
tf.truncated_normal_initializer()      将变量初始化为满足正态分布的随机值，但如果随机出来的值  主要参数为正态分布的均值和标准差
                                       偏离平均值超过2个标准差，那么这个数将会被重新随机
tf.Variable和tf.get_variable函数最大的区别在于指定变量名称的参数，对于tf.Variable函数，变量名称是一个可选的参数，通过name="v"的形式给出。
但是对于tf.get_variable函数，变量名称是一个必填的参数。tf.get_variable会根据这个名字去创建或者获取变量。           
'''

def variable_getvariable():
   v1 = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32), name="v1")
   v2 = tf.get_variable(shape=[2], initializer=tf.ones_initializer(), name="v2", dtype=tf.float32)
   with tf.Session() as sess:
       tf.global_variables_initializer().run()
       print("v1 is==>", sess.run(v1))
       print("v2 is==>", sess.run(v2))
       print(sess.run(tf.equal(v1, v2)))

'''
1.如果需要通过tf.get_variable获取一个已经创建的变量，需要通过tf.variable_scope函数来生成一个上下文管理器，并明确指定在这个上下文管理器中，
tf.get_variable将直接获取已经生成的变量。
2.当tf.variable_scope函数使用参数reuse=False或者reuse=None创建上下文管理器时，tf.get_variable会创建一个变量。如果同名的变量已经存在，则会报错。
3.当tf.variable_scope函数使用参数reuse=True创建上下文管理器时，tf.get_variable会获取一个变量。如果变量不存在，则会报错。
'''
def variable_scope_test():
    #在名字为f00的命名空间内创建名字为v的变量
    with tf.variable_scope("foo", reuse=False):
        v = tf.get_variable(name="v1", initializer=tf.constant_initializer(1.0), shape=[2])

    #当reuse=False或者reuse参数不填时，tf.get_variable为创建一个变量。刚已经在foo命名空间创建过v1变量。所以下面再次创建时会报错
    with tf.variable_scope("foo"):
        try:
             v = tf.get_variable(name="v1", initializer=tf.constant_initializer(1.0), shape=[2])
        except Exception:
             print("二次创建V1变量失败！")

    #当reuse=True时，tf.get_variable为获取变量
    with tf.variable_scope("foo", reuse=True):
        v1 = tf.get_variable(name="v1", shape=[2])
        print(v == v1)

    #因为haha命名空间没有创建v1变量，所以下面获取变量时会报错
    with tf.variable_scope("haha", reuse=True):
        try:
             v = tf.get_variable(name="v1", shape=[2])
        except Exception:
             print("获取变量失败！")

#Tensorflow中tf.variable_scope函数是可以嵌套的，下面的程序说明了当tf.variable_scope函数嵌套时，reuse参数的取值时如何确定的
def scope__nest():
    with tf.variable_scope("root"):
        #可以通过tf.get_variable_scope().reuse获取当前上下文管理器中的reuse参数的值
        print(tf.get_variable_scope().reuse)
        #新建一个嵌套的上下文管理器，并指定reuse为True
        with tf.variable_scope("foo", reuse=True):
            #输出True
            print(tf.get_variable_scope().reuse)
            #新建一个嵌套的上下文管理器但不指定reuse，这是reuse的取值会和最外层保持一致
            with tf.variable_scope("bar"):
                #输出True
                print(tf.get_variable_scope().reuse)
        #输出False，退出reuse设置为True的上下文之后reuse的值又回到了False
        print(tf.get_variable_scope().reuse)
    pass

#tf.variable_scope函数生成的上下文管理器也会创建一个Tensorflow中的命名空间，在命名空间内创建的变量名称都会带上这个命名空间名作为前缀。
#所以，tf.variable_scope函数除了可以控制tf.get_variable执行的功能，这个函数也提供了一个管理变量命名空间的方式。
def scope_variable_name():
    v1 = tf.get_variable("v", [1])
    print(v1.name)

    with tf.variable_scope("foo"):
        v2 = tf.get_variable("v", [1])
        print(v2.name)

    with tf.variable_scope("foo"):
        with tf.variable_scope("bar"):
            v3 = tf.get_variable("v", [1])
            print(v3.name)
        v4 = tf.get_variable("v1", [1])
        print(v4.name)
    #设置一个名称为空的命名空间，并设置reuse=True
    with tf.variable_scope("", reuse=True):
        #可以直接通过带命名空间名称的变量名来获取其他命名空间下的变量。
        v5 = tf.get_variable("foo/bar/v", [1])
        print(v5 == v3)
        v6 = tf.get_variable("foo/v", [1])
        print(v6 == v2)

if __name__ == "__main__":
    # variable_getvariable()
    # variable_scope_test()
    # scope__nest()
    scope_variable_name()
import tensorflow as tf

'''
tf.get_default_graph() -->是默认计算图。在Tensorflow程序中，系统会自动维护一个默认的计算图。
a.graph -->获取张量的计算图。a为计算图上面的一个节点
因为a、b张量没有特意指定计算图，所以他们都属于默认计算图上面的节点。所以张量计算图等于默认计算图
每一次运行程序默认计算图tf.get_default_graph()的值都会变化。但是不管怎么变张量计算图等于默认计算图
不同python脚本的默认计算图的值是不一样的。但是同一个脚本获取的值都是一样的
'''
def tensor():
    a = tf.constant([1.0, 2.0], name="a")
    b = tf.constant([2.0, 3.0], name="b")
    sess = tf.Session()
    print(sess.run(a))
    sess.close()
    result = a + b
    print(result)
    print(b.graph)
    print(a.graph)
    print(a.name)
    print(tf.get_default_graph())
    pass

'''
除了使用默认计算图。还可以利用tf.Graph()函数来生成新的计算图。不同计算图上的张量和运算都不会共享
张量情况下：1.在两个计算图上定义了相同张量名的张量，在不指定计算图的情况下，获取的是第二个计算图定义的张量
2.如果在两个计算图上定义了相同张量名v的张量，在指定g2计算图的情况下，获取张量v报错，错误信息显示v不在此g1计算图内
3.在一个指定的计算图里面获取其他计算图的张量，如果不用sess可以获取，用sess.run()获取则会报错不在此计算图内
'''
def tensor_graph():
    g1 = tf.Graph()
    with g1.as_default():
        v1 = tf.constant([1.0, 2.0 , 3.0], name="v1")
        print(v1.graph)

    g2 = tf.Graph()
    with g2.as_default():
        v = tf.constant([2.0, 3.0], name="v")
        print(v.graph)

    g3 = tf.Graph()
    with g3.as_default():
        v = tf.constant([2.0, 3.0], name="v")
        print(v.graph)

    print(v)
    print(v.graph)

    print(v1)
    print(v1.graph)

    with tf.Session(graph=g2) as sess:
        print(v1)
        print(v1.graph)
        try:
          print(sess.run(v))
          print(v.graph)
        except Exception:
            print('no graph!')
    pass

def variable_graph():
    #在g4计算图上面定义v变量，并初始值为0，shape为必填的参数
    g4 = tf.Graph()
    with g4.as_default():
        v = tf.get_variable("v", initializer=tf.zeros_initializer(), shape=[1])

    # 在g5计算图上面定义v变量，并初始值为0,shape为必填的参数
    g5 = tf.Graph()
    with g5.as_default():
        v = tf.get_variable("v", initializer=tf.ones_initializer(), shape=[1])

    #获取g4计算图上面的v变量
    with tf.Session(graph=g4) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        with tf.variable_scope("", reuse=True):
            print(sess.run(tf.get_variable("v")))

    # 获取g4计算图上面的v变量
    with tf.Session(graph=g5) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        with tf.variable_scope("", reuse=True):
            print(sess.run(tf.get_variable("v")))
    pass

'''
所有数据都是通过张量的形式来表示。从功能的角度来看，张量可以被简单的理解为多维数组。其中零阶张量表示标量(scalar)，也就是一个数
第一阶张量为向量(vector)，也就是一个一维数组，第n阶的张量可以理解为一个n维数组。但张量在Tensorflow中的实现并不是直接采用数组的形式，
它只是对Tensorflow中运算结果的引用。在张量中并没有真正的保存数字，他保存的是如何得到这些数字的计算过程。
tf.add的输出结果result为Tensor("add:0", shape=(2,), dtype=float32)，里面的三种属性分别为名字(name),维度(shape),类型(type).shape=(2,)表示是一个一维数组，长度为2
定义张量如果不指定类型的话会采用默认类型，比如不带小数点的数会被默认为int32，带小数点的会默认为float32,因为使用默认可能会导致潜在的类型不匹配问题，所以一般建议通过指定dtype来给出变量或者常量的类型。
比如下面的a+b因为类型相同所以不会报错。但是a+c因为类型不同，会报错。但是变量d指定了类型在计算时就不会报错了
Tensorflow支持14种类型，主要包括实数(tf.float32、tf.float64)、整数(tf.int8、tf.int16、tf.int32、tf.int64、tf.uint8)、布尔型(tf.bool)和复数(tf.complex64、tf.complex128)
Tensorflow张量使用主要总结为两大类:1.对中间计算结果的引用，比如如下a和b是结果result的中间结果。2.当计算图构造完成后，张量可以用来获取计算结果，也就是得到真实的数字，比如如下sess_result
'''
def add_tensor():
    #tf.constant是一个计算，这个计算结果为一个张量，保存在变量a中
    a = tf.constant([1.0, 2.0], name="a")
    b = tf.constant([[1.0, 2.0],  [2.0, 3.0]], name="b")
    c = tf.constant([5, 6], name="c")
    d = tf.constant([7, 8], name="d", dtype=tf.float32)
    result = tf.add(a, b, name="add")
    try:
        result_error = tf.add(a, c, name="add_error")
        print(result_error)
    except Exception:
        print("add error")
    result_type = tf.add(a, d, name="add_type")
    print(result_type)
    print(result)

    with tf.Session() as sess:
        sess_result = sess.run(result)
        print(sess_result)

if __name__ == "__main__":
    # tensor()
    # tensor_graph()
    # variable_graph()
    add_tensor()
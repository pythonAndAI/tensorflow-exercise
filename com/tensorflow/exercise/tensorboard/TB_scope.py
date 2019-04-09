import tensorflow as tf
from com.utils import Cmd_Util
'''
讲解了tf.variable_scope和tf.name_scope的区别。
相同点：都有管理命名空间的功能。
不同点：tf.variable_scope管理了tf.get_variable函数定义的变量的命名空间。而tf.name_scope对tf.get_variablle没有影响。
        tf.name_scope管理了tf.Variable函数生成的变量。
'''
def scope_diff():
    with tf.variable_scope("foo"):
        #在命名空间foo下获取变量bar，于是得到的变量名称为foo/bar
        a = tf.get_variable("bar", [1])
        print(a.name)

    with tf.variable_scope("bar"):
        #在命名空间bar下获取变量bar，于是得到的变量名称为bar/bar。此时变量bar/bar和foo/bar并不冲突，可以正常运行
        b = tf.get_variable("bar", [1])
        print(b.name)

    with tf.name_scope("a"):
        #使用tf.Variable函数生成的变量会受tf.name_scope的影响，于是这个变量的名称为a/variable
        a = tf.Variable([1])
        print(a.name)
        #tf.get_variable函数不受tf.name_scope函数的影响，于是变量并不在a这个命名空间中。于是输出b:0
        a = tf.get_variable("b", [1])
        print(a.name)

    with tf.name_scope("b"):
        #因为tf.get_variable函数不受tf.name_scope函数的影响，所以这里将试图获取名称为a的变量。然而这个变量已经声明了，于是会报错
        try:
            a = tf.get_variable("b", [1])
        except Exception:
            print("a variable error!")

#通过对命名空间管理，改进向量相加的代码，使得可视化得到的效果图更加清晰。
def scope_add_pre_TB():
    #将输入和输出定义放入各自的命名空间中，从而是的RensorBoard可以根据命名空间来整理可视化效果图上的节点
    with tf.name_scope("input1"):
        input1 = tf.constant([1.0, 2.0, 3.0], name="input1")

    with tf.name_scope("input2"):
        input2 = tf.Variable(tf.random_uniform([3]), name="input2")
        # input2 = tf.get_variable(name="inputs", initializer=tf.truncated_normal_initializer(stddev=0.1), shape=[3])

    with tf.name_scope("output"):
        output = tf.add_n([input1, input2], name="add")

    path = "E:\\Alls\\software\\tensorboard"
    writer = tf.summary.FileWriter(path, tf.get_default_graph())
    writer.close()

if __name__ == "__main__":
    # scope_diff()
    scope_add_pre_TB()
    Cmd_Util.run_tensorboard()
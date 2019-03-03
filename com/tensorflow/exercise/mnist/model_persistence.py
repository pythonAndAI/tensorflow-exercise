import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
#模型持久化，就是保存训练好的模型和还原保存的模型以便再次使用

'''
1.如下代码实现了持久化一个简单的Tensorflow模型的功能，在这段代码中，通过saver.save函数将模型保存到E:\Alls\软件\model\save\model.ckpt文件中。
Tensorflow模型一般会存在.ckpt的文件中。虽然以上程序只指定了一个文件路径，但是在这个文件目录下会生成三个文件。这是因为Tensorflow会把计算图的结构和图上参数取值分开保存。
2.第一个文件model.ckpt.meta，他Tensorflow计算图的结构，可以简单理解为神经网络的网络结构。第二个文件为model.ckpt，这个文件中保存了Tensorflow程序中每一个变量的取值。
最后一个文件为checkpoint文件，这个文件中保存了一个目录下所有的模型文件列表。
'''
def model_save():
    v1 = tf.Variable(tf.constant(1.0, shape=[2]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[2]), name="v2")
    result = v1 + v2

    #声明tf.train.Saver()类用于保存模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.save(sess, "E:\Alls\软件\model\save\model.ckpt")
        print(sess.run(result))

#定义模型运算的方式加载模型
def load_with_operation():
    #使用和保存模型代码中一样的方式来声明变量
    v1 = tf.Variable(tf.constant(1.0, shape=[2]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[2]), name="v2")
    result = v1 + v2

    # 声明tf.train.Saver()类用于保存模型
    saver = tf.train.Saver()
    #为了保存或加载部分变量，在声明tf.train.Saver类时可以提供一个列表来指定需要保存或者加载的变量，比如在加载模型的代码中使用saver = tf.train.Saver([v1])
    #命令来构建tf.train.Saver类，那么只有变量v1会被加载进来。如果运行修改后只加载了v1的代码会得到变量未初始化的错误.因为v2没加载，所有v2在运行初始化之前是没有值的。
    # saver = tf.train.Saver([v1])

    with tf.Session() as sess:
        #加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
        saver.restore(sess, "E:\Alls\软件\model\save\model.ckpt")
        print(sess.run(result))

#如果不希望加载模型时重复定义图上的运算，也可以直接加载已经持久化的图
def load_without_operation():
    save = tf.train.import_meta_graph("E:\Alls\软件\model\save\model.ckpt.meta")
    with tf.Session() as sess:
        save.restore(sess, "E:\Alls\软件\model\save\model.ckpt")
        print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

#当保存是的变量名称和加载时定义的变量名称不相等时，需要在定义saver类时重命名变量名
def variable_rename():
    v1 = tf.Variable(tf.constant(1.0, shape=[2]), name="other_v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[2]), name="other_v2")
    result = v1 + v2
    #在这个程序中对v1和v2的名称进行了修改，如果直接加载会报变量找不到的错误。为了解决这个问题，Tensorflow可以通过字典将模型保存时的变量名和需要加载加载的变量联系起来。
    saver = tf.train.Saver({"v1": v1, "v2": v2})

    with tf.Session() as sess:
        #加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
        saver.restore(sess, "E:\Alls\软件\model\save\model.ckpt")
        print(sess.run(result))

#以下为一个保存滑动平均模型的样例
def moving_average_model_save():
    v = tf.Variable(0, dtype=tf.float32, name="v")
    #在没有声明滑动平均模型时只有一个变量，所以以下语句只会输出“v:0”.
    for variable in tf.global_variables():
        print(variable.name)

    ema = tf.train.ExponentialMovingAverage(0.99)
    #在申明滑动平均模型之后，Tensorflow会自动生成一个影子变量
    maintain_average_op = ema.apply(tf.global_variables())
    for variable in tf.global_variables():
        print(variable.name)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(tf.assign(v, 10))
        sess.run(maintain_average_op)
        #保存时，Tensorflow会将v:0和v:ExponentialMovingAverage:0两个变量都存下来
        saver.save(sess, "E:\Alls\软件\model\save\model.ckpt")
        print(sess.run(v))
        print(sess.run(ema.average(v)))

#如下给出了如何通过变量重命名直接读取变量的滑动平均值，读取变量v的值实际上是上面代码中变量v的滑动平均值
#通过这个方法，就可以使用完全一样的代码来计算滑动平均模型的前向传播的结果
def moving_average_model_restore():
    v = tf.Variable(0, dtype=tf.float32, name="v")

    #1.通过saver = tf.train.Saver()方式创建的saver类，读取的是变量v的值10
    # saver = tf.train.Saver()
    #2.通过如下变量重命名将原来v的滑动平均值直接赋值给v，所以v的值为0.099999905
    # saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
    #3.为了方便加载时重命名滑动平均变量，tf.train.ExponentialMovingAverage提供了variables_to_restore()函数来生成tf.train.Saver类所需要的变量重命名字典。
    #此方式和2的方式输出的都是滑动平均值，作用是相等的
    eam = tf.train.ExponentialMovingAverage(0.99)
    saver = tf.train.Saver(eam.variables_to_restore())

    with tf.Session() as sess:
        saver.restore(sess, "E:\Alls\软件\model\save\model.ckpt")
        print(sess.run(v))

'''
使用tf.train.Saver会保存运行Tensorflow程序所需要的全部信息，然而有时并不需要某些信息。比如在测试或者离线预测时，只需要知道如何从神经网络的输入层经过前向传播
计算得到输出层即可，而不需要类似于变量初始化、模型保存等辅助节点的信息。而且，将变量取值和计算图结构分成不同的文件存储有时候也不方便，于是Tensorflow提供了
convert_variables_to_constants函数，通过这个函数可以将计算图中的变量及其取值通过常量的方式保存，这样整个Tensorflow计算图可以统一存放在一个文件中。
'''
def model_Save_a_file():
    v1 = tf.Variable(tf.constant(1.0, shape=[2]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[2]), name="v2")
    result = v1 + v2

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
        graph_def = tf.get_default_graph().as_graph_def()
        #将图中的变量及其取值转换为常量，同时将图中不必要的节点去掉。如果只关心程序中定义的某些计算时，和这些计算无关的节点就没有必要导出并保存了。
        #在下面一行代码中，最后一个参数['add']给出了需要保存的节点名称。add节点是上面定义的两个变量相加的操作。注意这里给出的是计算节点的名称，所以没有后面的:0。
        out_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
        #将导出的模型存入文件
        with tf.gfile.GFile("E:\Alls\软件\model\save\combined_model_pb", "wb") as f:
            f.write(out_graph_def.SerializeToString())
        print(sess.run(result))

#通过以下程序可以直接计算定义的加法运算结果。当只需要得到计算图中某个节点的取值时，这提供了一个更加方便的方法。第六章将使用这种方法来使用训练好的模型完成迁移学习
def read_save_a_file_model():
    with tf.Session() as sess:
        model_path = "E:\Alls\软件\model\save\combined_model_pb"
        with gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        #将graph_def中保存的图加载到当前的图中，return_elements = ["add:0"]给出了返回的张量的名称。在保存的时候给出的是计算节点的名称，所以为"add"。
        #在加载的时候给出的是张量的名称，所以是add:0
        result = tf.import_graph_def(graph_def, return_elements=["add:0"])
        print(sess.run(result))


if __name__ == "__main__":
    # model_save()
    # load_with_operation()
    # load_without_operation()
    # variable_rename()
    # moving_average_model_save()
    # moving_average_model_restore()
    # model_Save_a_file()
    read_save_a_file_model()
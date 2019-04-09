import tensorflow as tf
from com.utils import Cmd_Util
from com.utils import File_Util

#定义一个简单的计算图，实现向量加法的操作
input1 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32, name="input1")
input2 = tf.Variable(tf.random_uniform([3], dtype=tf.float32), name="input2")

output = tf.add_n([input1, input2], name="add")

path = File_Util.get_path("E:\\Alls\\software\\tensorboard\\aa")
writer = tf.summary.FileWriter(path, tf.get_default_graph())
writer.close()
Cmd_Util.run_tensorboard(path=path)
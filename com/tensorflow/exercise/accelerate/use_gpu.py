import tensorflow as tf

'''
TensorFlow程序可以通过tf.device函数来指定运行每一个操作的设备，这个设备可以是本地的CPU或者GPU，也可以是某一台远程的服务器。TensorFlow会给每一个可用的设备一个名称，
tf.device函数可以通过设备的名称来指定执行运算的设备。比如CPU在TensorFlow中的名称为/cpu:0。在默认情况下，即使机器有多个CPU，TensorFlow也不会区分它们，所有的CPU都使用
/cpu:0作为名称。而一台机器上不同GPU的名称是不同的，第n个GPU在TensorFlow中的名称为/gpu:n。比如第一个GPU的名称为/gpu:0，第二个GPU名称为/gpu:1，以此类推。
TensorFlow提供了一个快捷的方式来查看运行每一个运算的设备。在生成会话时，可以通过设备log_device_placement参数来打印每一个运算的设备。
'''
#输出运算设备
def print_arithmetic_device():
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name="a")
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name="b")
    c = a + b
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    #输出a: (Const): /job:localhost/replica:0/task:0/device:CPU:0。表示是通过CPU运行的，因为它的设备名称中包含了/cpu:0
    #在配置好GPU环境的TensorFlow中，如果操作没有明确指定运行设备，则TensorFlow会优先选择GPU
    print(sess.run(c))
    sess.close()

'''
1.指定运算设备：如果需要将某些运算放到不同的GPU或者CPU上，就需要通过tf.device来手工指定。
2.以下代码可以看到生成常量a和b的操作被加载到了CPU上，而add操作被放到了第二个GPU“/gpu:1”上。在TensorFlow中，不是所有的操作都可以被放到GPU上，如果强行将无法放在GPU上的
操作指定到GPU上，那么程序会报错。
3.不同版本的TensorFlow对GPU的支持不一样，如果程序中全部使用强制指定设备的方式会降低程序的可移植性。在TensorFlow的kernel中定义了那些操作可以跑在GPU上。
参考：https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/kernels/目录下。
'''
def designated_computing_device():
    with tf.device("/cpu:0"):
        a = tf.constant([1.0, 2.0, 3.0], shape=[3], name="a")
        b = tf.constant([1.0, 2.0, 3.0], shape=[3], name="b")
    #如果有GPU的话，需要指定
    with tf.device("/gpu:1"):
        c = a + b
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    # 输出a: (Const): /job:localhost/replica:0/task:0/device:CPU:0。表示是通过CPU运行的，因为它的设备名称中包含了/cpu:0
    # 在配置好GPU环境的TensorFlow中，如果操作没有明确指定运行设备，则TensorFlow会优先选择GPU
    print(sess.run(c))
    sess.close()

'''
为了避免指定GPU的运算报错，TensorFlow在生成会话时可以指定allow_soft_placement参数。当allow_soft_placement参数设置为True时，如果运算无法由GPU执行，
那么TensorFlow会自动将它放到CPU上执行。
'''
def automatic_computing_device():
    with tf.device("/gpu:0"):
        a_gpu = tf.Variable(0, name="a_gpu")
    #通过allow_soft_placement参数自动将无法放到GPU上的操作放回CPU上
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    sess.run(a_gpu)

'''
1.虽然GPU可以加速TensorFlow的计算，但一般来说不会把所有的操作全部放到GPU上。一个比较好的实践是将计算密集型的运算放在GPU上，而把其他操作放到CPU上。GPU是计算中相对
独立的资源，将计算放入或者转出GPU都需要额外的时间。而且GPU需要将计算时用到的数据从内存复制到GPU设备上，这也需要额外的时间。TensorFlow可以自动完成这些操作而不需要
用户特别处理，但为了提高程序运行的速度，用户也需要尽量将相关的运算放到同一个设备上。
2.TensorFlow默认会占用设备上的所有GPU以及每个GPU的所有显存。如果在一个TensorFlow程序中只需要使用部分GPU，可以通过设置CUDA_VISIBLE_DEVICES环境变量来控制。
2.1.只使用第二块GPU(GPU编号从0开始)。在demo_code.py中，机器上的第二块GPU的名称变成/gpu:0，不过在运行时所有/gpu:0的运算将被放在第二块GPU上。
CUDA_VISIBLE_DEVICES=1 demo_code.py
#只使用第一块和第二块GPU
CUDA_VISIBLE_DEVICES=0,1 demo_code.py
2.2.TensorFlow也支持在程序中设置环境变量，如下：
import os
#只使用第三块GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
'''

'''
虽然TensorFlow默认会一次性占用一个GPU的所有显存，但是TensorFlow也支持动态分配GPU的显存，使得一块GPU上可以同时运行多个任务。
'''
def assign_graphics_card():
    config = tf.ConfigProto()

    #让TensorFlow按需分配显存
    config.gpu_options.allow_growth = True

    #或者直接按固定的比例分配。以下代码会占用所有可使用GPU的40%显存
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)

if __name__ == "__main__":
    # print_arithmetic_device()
    designated_computing_device()
    # automatic_computing_device()
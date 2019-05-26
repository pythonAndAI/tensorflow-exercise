import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from com.tensorflow.exercise.mnist.final import mnist_inference
from com.utils import constant_Util

#以下代码实现了同步模式的分布式神经网络训练过程。

#配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99

#模型保存在路径
MODEL_SAVE_PATH = "E:\Alls\软件\model\save"

#通过flags指定运行的参数。distributed_basis.py中对于不同的任务(task)给出了不同的程序，但这不是一种可扩展的方式。在下面将使用运行程序时给出的参数来配置在不同任务中运行的程序。
FLAGS = tf.app.flags.FLAGS

#指定当前运行的是参数服务器还是计算服务器。参数服务器只负责TensorFlow中变量的维护和管理，计算服务器负责每一轮迭代时运行反向传播过程。
tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
#指定集群中的参数服务器地址
tf.app.flags.DEFINE_string('ps_hosts', 'tf-ps0:2222,tf-ps1:1111',
                           'Comma-separated list of hostname:port for the parameter server '
                           'jobs. e.g. "tf-ps0:2222,tf-ps1:1111" ')
#指定集群中的计算服务器地址
tf.app.flags.DEFINE_string('worker_hosts', 'tf-worker0:2222,tf-worker1:1111',
                           'Comma-separated list of hostname:port for the worker jobs. '
                           'jobs. e.g. "tf-worker0:2222,tf-worker1:1111" ')
#指定当前程序的任务ID。TensorFlow会自动根据参数服务器/计算服务器列表中的端口号来启东服务。注意参数服务器和计算服务器的编号都是从0开始的。
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

#和异步模式类似的定义TensorFlow的计算图，并返回每一轮迭代时需要运行的操作。唯一的区别在于使用tf.train.SyncReplicasOptimizer函数处理同步更新。
def build_model(x, y_, n_workers, is_chief):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.contrib.framework.get_or_create_global_step()
    #计算损失函数并定义反向传播过程
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE, LEARNING_RATE_DECAY)

    #通过tf.train.SyncReplicasOptimizer函数实现同步更新
    opt = tf.train.SyncReplicasOptimizer(tf.train.GradientDescentOptimizer(learning_rate), replicas_to_aggregate=n_workers, total_num_replicas=n_workers)
    sync_replicas_hook = opt.make_session_run_hook(is_chief)
    train_op = opt.minimize(loss, global_step=global_step)

    #定义每一轮迭代需要运行的操作
    if is_chief:
        #定义变量的滑动平均值
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([variable_averages_op, train_op]):
            train_op = tf.no_op()
    return global_step, loss, train_op, sync_replicas_hook

def main(argv=None):
    #解析flags并通过tf.train.ClusterSpec配置TensorFlow集群
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    n_workers = len(worker_hosts)
    cluster = tf.train.ClusterSpec({"ps" : ps_hosts, "workers" : worker_hosts})
    #通过tf.train.ClusterSpec以及当前任务创建tf.train.Server
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)
    #参数服务器只需要管理TensorFlow中的变量，不需要执行训练的过程。server.join()会一直停在这条语句上。
    if FLAGS.job_name == "ps":
        with tf.device("/cpu:0"):
            server.join()
    #定义计算服务器需要运行的操作
    is_chief = (FLAGS.task_id == 0)
    mnist = input_data.read_data_sets(constant_Util.MNIST_PATH, one_hot=True)

    #通过tf.train.replica_device_setter函数来指定执行每一个运算的设备。tf.train.replica_device_setter函数会自动将所有的参数分配到参数服务器上，将计算分配到当前的计算服务器上。
    device_setter = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_id, cluster=cluster)

    with tf.device(device_setter):
        #定义输入并得到每一轮迭代需要运行的操作
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODES], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODES], name='y-input')
        global_step, loss, train_op, sync_replicas_hook = build_model(x, y_, is_chief)

        #把处理同步更新的hook也加进来。
        hooks = [sync_replicas_hook, tf.train.StopAtStepHook(last_step=TRAINING_STEPS)]
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        #通过tf.train.MonitoredTrainingSession管理训练深度学习模型的通用功能
        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, checkpoint_dir=MODEL_SAVE_PATH, hooks=hooks, save_checkpoint_secs=60, config=sess_config) as mon_sess:
            print("session started.")
            step = 0
            start_time = time.time()

            #执行迭代过程。在迭代过程中tf.train.MonitoredTrainingSession会帮助完成初始化、从checkpoint中加载训练过的模型、输出日志并保存模型，所以以下程序中不需要再调用这些过程。
            #tf.train.StopAtStepHook会帮忙判断是否需要退出。
            while not mon_sess.should_stop():
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                _, loss_value, global_step_value = mon_sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

                #每隔一段时间输出训练信息。不同的计算服务器都会更新全局的训练轮数，所以这里使用global_step_value得到在训练中使用过的batch的总数。
                if step > 0 and step % 100 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / global_step_value
                    format_str = "After %d training steps %d global steps, loss on training batch is %g. (%.3f sec/batch)"
                    print(format_str % (step, global_step_value, loss_value, sec_per_batch))
                step += 1

if __name__ == "__main__":
    tf.app.run()

'''
以上程序要启动一个拥有一个参数服务器、两个计算服务器的集群。
1. 首先需要先在运行参数服务器的机器上启动以下命令:
python distributen_asynchronous.py --job_name='ps' --task_id=0 -- ps_hosts='tf.ps0:2222' --worker_hosts='tf-worker0:2222,tf-worker1:1111'
2. 然后在运行第一个计算服务器的机器上启动以下命令:
python distributen_asynchronous.py --job_name='worker' --task_id=0 -- ps_hosts='tf.ps0:2222' --worker_hosts='tf-worker0:2222,tf-worker1:1111'
3. 最后在运行第二个计算服务器的机器上启动以下命令:
python distributen_asynchronous.py --job_name='worker' --task_id=1 -- ps_hosts='tf.ps0:2222' --worker_hosts='tf-worker0:2222,tf-worker1:1111'

在启动第一个计算服务器之后，这个计算服务器就会尝试连接其他的服务器(包括计算服务器和参数服务器)。如果其他服务器还没有启动，
则被启动的计算服务器会提示等待连接其他服务器，以下代码展示了一个预警信息：
tensorflow\core\distributed_runtime\master.cc:221] CreateSession still waiting for response from worker: /job:worker/replica:0/task:1
不过这不会影响TensorFlow集群的启动。当TensorFlow集群中所有服务器都被启动之后，每一个计算服务器将不再警告。在TensorFlow集群完全启动之后，训练过程将被执行。

和异步模式不同，在同步模式下，global_step差不多是两个计算服务器local_step的平均值。比如在第二个计算服务器还没有开始之前，global_step是第一个服务器local_step
的一半。这是因为同步模式要求收集replicas_to_aggregate份梯度才会开始更新(注意这里TensorFlow不要求每一份梯度来自不同的计算服务器)。同步模式不仅仅是一次使用多份
梯度，tf.train.SyncReplicasOptimizer的实现同时也保证了不会出现陈旧变量的问题。tf.train.SyncReplicasOptimizer函数会记录每一份梯度是不是由最新的变量值计算得到的，
如果不是，那么这一份梯度将会被丢弃。
'''
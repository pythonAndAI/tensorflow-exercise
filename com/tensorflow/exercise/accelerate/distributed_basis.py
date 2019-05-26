import tensorflow as tf

'''
分布式TensorFlow原理
通过多GPU并行的方式可以达到很好的加速效果。然而一台机器上能够安装的GPU有限，要进一步提升深度学习模型的训练速度，就需要将TensorFlow分布式运行在多台机器上。
'''

#以下代码展示了如何创建一个最简单的TensorFlow集群。只有一个任务的集群。
def create_local_cluster():
    c = tf.constant("Hello, distributed TensorFlow!")
    #创建一个本地TensorFlow集群
    server = tf.train.Server.create_local_server()
    #在集群上创建会话
    sess = tf.Session(server.target)
    #输出Hello, distributed TensorFlow!
    print(sess.run(c))

#当一个TensorFlow集群有多个任务时，需要使用tf.train.ClusterSpec来指定运行每一个任务的机器。通过tf.train.Server.target生成的会话可以统一管理整个TensorFlow集群中的资源。
def more_task_cluster1():
    c = tf.constant("Hello from server1!")
    #生成一个有两个任务的集群，一个任务跑在本地2222端口，另一个任务跑在本地2223端口
    cluster = tf.train.ClusterSpec({"local" : ["localhost:2222", "localhost:2223"]})
    #通过上面生成的集群配置生成Server，并通过job_name和task_index指定当前所启动的任务。因为该任务是第一个任务，所以task_index为0
    server = tf.train.Server(cluster, job_name="local", task_index=0)
    #通过server.target生成会话来使用TensorFlow集群中的资源。通过设置log_device_placement可以看到执行每一个操作的任务。
    sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
    print(sess.run(c))
    server.join()

#先执行more_task_cluster1方法，不关闭的情况下再次执行more_task_cluster2方法。不然就会一直打印等待task1
def more_task_cluster2(schedul=True):
    if schedul:
        #默认会同more_task_cluster1被放到/job:local/replica:0/task:0/device:CPU:0设备上。也就是说这个计算将由第一个任务来执行。
        c = tf.constant("Hello from server2!")
    else:
        #同使用多GPU类似，TensorFlow支持通过tf.device来指定操作运行在哪个任务上。比如如下定义就可以看到此计算将被调度到/job:local/replica:0/task:1/device:CPU:0上面。
        with tf.device("/job:local/task:1"):
            c = tf.constant("Hello from server2!")

    #生成一个有两个任务的集群，一个任务跑在本地2222端口，另一个任务跑在本地2223端口
    cluster = tf.train.ClusterSpec({"local" : ["localhost:2222", "localhost:2223"]})
    #通过上面生成的集群配置生成Server，并通过job_name和task_index指定当前所启动的任务。因为该任务是第二个任务，所以task_index为1
    server = tf.train.Server(cluster, job_name="local", task_index=1)
    #通过server.target生成会话来使用TensorFlow集群中的资源。通过设置log_device_placement可以看到执行每一个操作的任务。
    sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
    print(sess.run(c))
    server.join()

if __name__ == "__main__":
    # create_local_cluster()
    # more_task_cluster1()
    more_task_cluster2()

1.在以上distributed_basis.py样例中只定义了一个工作“local”。但一般在训练深度学习模型时，会定义两个工作。一个工作专门负责存储、获取以及更新变量的取值，这个工作所包含的任务统称为
参数服务器(parameter server, ps)。另外一个工作负责运行反向传播算法来获取参数梯度，这个工作所包含的任务统称为计算服务器(worker)。下面给出了一个比较常见的用于训练深度学习模型的
TensorFlow集群配置方法(tf-worker(i)和tf-ps(i)都是服务器地址。)：
tf.train.ClusterSpec({
    "worker": [
        "tf-worker0:2222",
        "tf-worker1:2222",
        "tf-worker2:2222"
    ],
    "ps": [
        "tf-ps0:2222",
        "tf-ps1:2222"
    ]
})

2.使用分布式TensorFlow训练深度学习模型一般有两种方式。
2.1 第一种方式叫做计算图内分布式(in-graph replication)。使用这种分布式训练方式时，所有的任务都会使用一个TensorFlow计算图中的变量(也就是深度学习模型中的参数)，而只是将计算部分
发布到不同的计算服务器上。案例mnist_multi_gpu_train.py给出的使用多GPU样例程序就是这种方式。多GPU样例程序将计算复制了多份，每一份放到一个GPU上运行计算。但不同的GPU使用的参数都
是在一个TensorFlow计算图中的。因为参数都是存在同一个计算图中，所以同步更新参数比较容易控制。mnist_multi_gpu_train.py中给出的代码也实现了参数的同步更新。然而因为计算图内分布式
需要有一个中心节点来生成这个计算图并分配计算任务，所以当数据量太大时，这个中心节点容易造成性能瓶颈。

2.2 另外一种分布式TensorFlow训练深度学习模型的方式叫计算图之间分布式(between-graph replication)。使用这种分布式方式时，在每一个计算服务器上都会创建一个独立的TensorFlow计算图，
但不同计算图中的相同参数需要以一种固定的方式放到同一个参数服务器上。TensorFlow提供了tf.train.replica_device_setter函数来帮助完成这一个过程，案例======。因为每个计算服务器的
TensorFlow计算图是独立的，所以这种方式的并行度要更高。但在计算图之间分布式下进行参数的同步更新比较困难。为了解决这个问题，TensorFlow提供了tf.train.SyncReplicasOptimizer函数
来帮助实现参数的同步更新。这让计算图之间分布式方式被更加广泛地使用。

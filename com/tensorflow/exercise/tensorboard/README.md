Tensorborad--> 是Tensorflow的可视化工具，它可以通过Tensorflow程序运行过程中输出的日志文件可视化Tensorflow程序的运行状态。Tensorflow和Tensorborad程序跑在不同的进程中，
               Tensorboard会自动读取最新的Tensorflow日志文件，并呈现当前Tensorflow程序运行的最新状态。

一、Tensorboard简介
例如TB_basis.py案例，执行完程序之后，会在path路径底下生成一个文件，在cmd里面运行以下命令：tensorboard -- logdir=path (运行Tensorflow，并将日志的地址指向上面程序日志输出的地址path)
运行以上命令会启动一个服务(http://LAPTOP-PIG9M9R9:6006)，这个服务的端口默认为6006，使用--port参数可以改变启动服务的端口。通过浏览器打开http://LAPTOP-PIG9M9R9:6006。会看到一个界面，
在界面的上方，展示的内容是“GRAPHS”，表示图中可视化的内容是Tensorflow的计算图。打开界面会默认进入GRAPHS界面，在该界面中可以看到上面程序Tensorflow计算图的可视化结果。另外有一个“INACTIVE”
选项，点开这个选项可以看到Tensorboard能够可视化的其他内容。比如：IMAGES、AUDIO等。

二、TensorFlow计算图可视化
1. Tensorboard可视化得到的图不仅是将TensorFlow计算图中的节点和边直接可视化，它会根据每个TensorFlow计算节点的命名空间来整理可视化得到的效果图，使得神经网络的整体结构不会被过多的细节所淹没。
除了TensorFlow计算图的结构。Tensorboard还可以展示TensorFlow计算节点上的其他信息。

2.为了更好地组织可视化效果图上的节点。Tensorboard支持通过TensorFlow命名空间来整理可视化效果图上的节点。在Tensorboard的默认视图中，TensorFlow计算图中同一个命名空间下的所有节点会被缩略成一个节点，
只有顶层命名空间中的节点才会被显示在Tensorboard可视化效果图上。Tensorboard通过tf.name_scope函数来管理命名空间。tf.name_scope函数不会影响tf.get_variable()函数定义的变量的命名空间。参考案例TB_scope.py

参考：https://www.cnblogs.com/bestExpert/p/10678844.html
import tensorflow as tf
import numpy as np
import time
import threading
'''
队列和多线程
虽然使用这些图像数据预处理的方法可以减少无关因素对图像识别模型效果的影响，但这些复杂的预处理过程也会减慢整个训练过程。
为了避免图像预处理成为神经网络模型训练效率的瓶颈，Tensorflow提供了一套多线程处理输入数据的框架。
============================
指定原始数据的文件列表---->创建文件列表队列---->从文件中读取数据---->数据预处理---->整理成batch作为神经网络输入
'''

#在Tensorflow中，队列和变量类似，都是计算图上有状态的节点。修改队列状态的操作有Enqueue、EnqueueMany、Dequeue
def queue_test():
    #创建一个先进先出队列，指定队列中最多可以保存两个元素，并指定类型为整数。除了FIFOQueue队列，还有tf.RandomShuffleQueue队列，
    # RandomShuffleQueue会将队列中的元素打乱，每次出队列操作得到的是从当前队列所有元素中随机选择的一个。在训练神经网络时希望每次使用的训练数据尽量随机，RandomShuffleQueue就提供了这个功能。
    q = tf.FIFOQueue(2, "int32")
    #初始化队列元素。和变量初始化一样，在使用队列之前需要明确的调用这个初始化过程
    init = q.enqueue_many(([0, 10], ))
    #将队列中的第一个元素出列
    x = q.dequeue()
    #将得到的值加1
    y = x + 1
    #将加1后的值重新加入队列
    q_inc = q.enqueue([y])

    with tf.Session() as sess:
        #运行初始化队列的操作
        init.run()
        for i in range(5):
            #运行q_inc将执行数据出队列、出对的元素加1、重新加入队列的整个过程
            v, _ = sess.run([x, q_inc])
            print(v)

#Tensorflow提供了tf.train.Coordinator和tf.train.QueueRunner两个类来完成多线程协同的功能，tf.train.Coordinator主要用于协同多个线程一起停止，并提供了should_stop、request_stop
#和join三个函数。在启动每个线程之前，需要先声明一个tf.train.Coordinator类并将这个类传入每一个创建的线程中。启动的线程需要一直查询tf.train.Coordinator类中提供的should_stop函数，
#当这个函数返回值为True时，则当前线程也需要退出。每一个启动的线程都可以通过调用request_stop函数来通知其他线程退出，当某一个线程调用了request_stop函数之后，should_stop函数的
#返回值将被设置为True，这样其他的线程就可以同时终止了
def coordinator_basis(coord, worker_id):
    #调用should_stop函数判断线程是否需要停止
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("stop form id is:", worker_id)
            #调用request_stop函数来通知其他线程停止
            coord.request_stop()
        else:
            print("work on id is:", worker_id)

        #暂停1秒
        time.sleep(1)

def coordinator_test():
    coord = tf.train.Coordinator()
    #声明创建5个线程
    threads = [threading.Thread(target=coordinator_basis, args=(coord, i)) for i in range(5)]
    #启动所有线程
    for t in threads:
        t.start()
    #等待所有线程退出
    coord.join(threads)
    print("all thread stop!")

#tf.train.QueueRunner主要用于启动多个线程来操作同一个队列，启动的这些线程可以通过上面介绍的tf.train.Coordinator类来统一管理
def queueRunner_test():
    queue = tf.FIFOQueue(100, "float")
    #定义队列的入队操作
    enqueue_op = queue.enqueue([tf.random_normal([1])])
    #使用tf.train.QueueRunner来创建多个线程运行队列的入队操作。第一个参数为被操作的队列，第二个参数表示启动的线程数，
    # 如下[enqueue_op] * 5表示启动5个线程，每个线程中运行的是enqueue_op操作。
    qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

    #将定义过得QueueRunner加入Tensorflow计算图上指定的集合。tf.train.add_queue_runner函数如果没有指定集合，则加入默认集合tf.GraphKeys.QUEUE_RUNNERS。
    # 下面的函数就是将刚刚定义的qr加入默认的tf.GraphKeys.QUEUE_RUNNERS集合
    tf.train.add_queue_runner(qr)
    #定义出队操作
    out_tensor = queue.dequeue()

    with tf.Session() as sess:
        #定义Coordinator来协同启动的线程
        coord = tf.train.Coordinator()
        #使用tf.train.QueueRunner时，需要明确调用tf.train.start_queue_runners来启动所有线程。否则因为没有线程运行入队操作，当调用出队操作时，程序会一直等待入队操作被运行。
        #tf.train.start_queue_runners函数会默认启动tf.GraphKeys.QUEUE_RUNNERS集合中所有的QueueRunner。因为这个函数只支持启动指定集合中的QueueRunner。
        #所以一般来说tf.train.add_queue_runner函数和tf.train.start_queue_runners函数会指定同一个集合。
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(sess.run(queue.size()))
        for i in range(3):
            print(sess.run(queue.size()))
            print(sess.run(out_tensor))
        coord.request_stop()
        coord.join(threads)
        print("all thread stop!")


if __name__ == "__main__":
    # queue_test()
    # coordinator_test()
    queueRunner_test()
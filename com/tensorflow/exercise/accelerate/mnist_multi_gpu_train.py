from datetime import datetime
import tensorflow as tf
import os
import time
from com.tensorflow.exercise.mnist.final import mnist_inference

#下面给出TensorFlow代码，在一台机器的多个GPU上并行训练深度学习模型。因为一般来说一台机器上的多个GPU性能相似，所以在这种设置下会更多地采用同步模式训练深度学习模型。
#通过调整参数N_GPU，可以实验同步模式下随着GPU个数的增加训练速度的加速比率，当使用两个GPU时，模型的训练速度是使用一个GPU的1.92倍。当GPU数量增加时，虽然加速比不再是线性，但
#TensorFlow仍然可以通过增加GPU数量有效地加速深度学习模型的训练过程。

#定义训练神经网络时需要用到的参数
BATCH_SIZE = 100
LEARNING_GATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.001
TRAINING_STEP = 1000
MOVING_AVERAGE_DECAY = 0.99
N_GPU = 2

#定义日志和模型输出的路径
MODEL_SAVE_PATH = "E:\Alls\软件\model\save"
MODEL_NAME = "model.ckpt"

#定义数据存储的路径。因为需要为不同的GPU提供不同的训练数据，所以通过placeholder的方式就需要手动准备多份数据。为了方便训练数据的获取过程，可以采用Dataset的方式从TFRecord中读取数据。
DATA_PATH = "output.tfrecords"

#定义输入队列得到训练数据
def get_input():
    dataset = tf.data.TFRecordDataset(DATA_PATH)
    #定义数据解析格式
    def parse(record):
        features = tf.parse_single_example(record, features={
            'image_raw' : tf.FixedLenFeature([], tf.string),
            'pixels' : tf.FixedLenFeature([], tf.int64),
            'label' : tf.FixedLenFeature([], tf.int64)
        })
        decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
        reshaped_image = tf.reshape(decoded_image, [784])
        retyped_image = tf.cast(reshaped_image, tf.float32)
        label = tf.cast(features["label"], tf.int32)

        return retyped_image, label
    dataset = dataset.map(parse).shuffle(buffer_size=10000).repeat(10).batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    features, label = iterator.get_next()
    return features, label

#定义损失函数。对于给定的训练数据、正则化损失计算规则和命名空间，计算在这个命名空间下的总损失。之所以需要给定命名空间是因为不同的GPU上计算得出的正则化损失都会加入名为loss
#的集合，如果不通过命名空间就会将不同GPU上的正则化损失都加进来。
def get_loss(x, y_, regularizer, scope, reuse_variable=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variable):
        y = mnist_inference.inference(x, regularizer)
    #计算交叉熵损失
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
    #计算当前GPU上计算得到的正则化损失
    regularization_loss = tf.add_n(tf.get_collection("loss", scope))
    #计算最终的总损失
    loss = cross_entropy + regularization_loss
    return loss

#计算每一个变量梯度的平均值
def average_gradients(tower_grads):
    average_grads = []

    #枚举所有的变量和变量在不同GPU上计算得出的梯度
    for grad_and_vars in zip(*tower_grads):
        #计算所有GPU上的梯度平均值
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        #将变量和它的平均梯度对应起来
        average_grads.append(grad_and_var)
    #返回所有变量的平均梯度，这个将被用于变量的更新
    return average_grads

#住训练过程
def main(argv=None):
    #将简单的运算放到CPU上，只有神经网络的训练过程放在GPU上
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        #定义基本的训练过程
        x, y_ = get_input()
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

        global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_GATE_BASE, global_step, 60000 / BATCH_SIZE, LEARNING_RATE_DECAY)
        opt = tf.train.GradientDescentOptimizer(learning_rate)

        tower_grads = []
        reuse_variables = False
        #将神经网络的优化过程跑在不同的GPU上
        for i in range(N_GPU):
            #将优化过程指定在一个GPU上
            with tf.device("/gpu:%d" % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    cur_loss = get_loss(x, y_, regularizer, scope, reuse_variables)
                    #将第一次声明变量之后，将控制变量重用的参数设置为True。这样可以让不同的GPU更新同一组参数。
                    reuse_variables = True
                    #这是第一部分minimize()。它返回（梯度，变量）对的列表
                    grads = opt.compute_gradients(cur_loss)
                    tower_grads.append(grads)
        #计算变量的平均梯度
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)

        #使用平均梯度更新参数。这是第二部分minimize()。它返回一个Operation应用渐变的。
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        #计算变量的滑动平均值
        variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = (tf.moving_average_variables() + tf.trainable_variables())
        variables_average_op = variable_average.apply(variables_to_average)
        #每一轮迭代需要更新变量的取值并更新变量的滑动平均值。
        train_op = tf.group(apply_gradient_op, variables_average_op)

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            #初始化所有变量并启动队列
            init.run()
            summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)
            for step in range(TRAINING_STEP):
                #执行神经网络训练操作，并记录训练操作的运行时间
                start_time = time.time()
                _, loss_value = sess.run([train_op, cur_loss])
                duration = time.time() - start_time

                #每隔一段时间输出当前的训练进度，并统计训练速度
                if step != 0 and step % 10 == 0:
                    #计算当前step使用过的训练数据个数。因为在每一次运行训练操作时，每一个GPU都会使用一个batch的训练数据，所以总共用到的训练数据个数为batch大小 * GPU个数
                    num_examples_per_step = BATCH_SIZE * N_GPU

                    #num_examples_per_step为本次迭代使用到的训练数据个数，duration为运行当前训练过程使用的时间，于是平均每秒可以处理的训练数据个数为num_examples_per_step / duration
                    examples_per_sec = num_examples_per_step / duration

                    #duration为运行当前训练过程使用的时间，因为在每一个训练过程中，每一个GPU都会使用一个batch的训练数据，所以在单个batch上的训练所需要时间为duration / N_GPU
                    sec_per_batch = duration / N_GPU

                    #输出训练信息
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                    #通过TensoraBoard可视化训练模型
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, step)

                #每个一段时间保存当前的模型
                if step % 1000 == 0 or (step + 1) == TRAINING_STEP:
                    checkpoint_patch = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                    saver.save(sess, checkpoint_patch, global_step=step)

if __name__ == "__main__":
    tf.app.run()


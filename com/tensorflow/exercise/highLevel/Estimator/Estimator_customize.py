import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#若使用预先定义好的模型，除了不能灵活选择模型的结构，模型使用的损失函数和每一层使用的激活函数等也都是预先定义好的。为了更加灵活地构建模型，Estimator支持使用自定义的模型结构。
#从以下代码可以看出，Estimator能非常好地支持自定义模型，而且模型结构的定义过程中也可以使用其他的Tensorflow高层封装（比如代码中使用到的tf.layers）。Estimator在支持自定义模型结构的同时，并不影响它对训练过程的封装。

#通过tf.layers来定义模型结构。这里可以使用原生态Tensorflow API或者任何Tensorflow的高层封装。X给出了输入层张量，is_training指明了是否为训练。该函数返回前向传播的结果
def lenet(X, is_training):
    #将输入转化为卷积层需要的形状
    x = tf.reshape(X, shape=[-1, 28, 28, 1])

    net = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 1024)
    net = tf.layers.dropout(net, rate=0.4, training=is_training)
    return tf.layers.dense(net, 10)

#自定义Estimator中使用的模型。定义的函数有4个输入，features给出了再输入函数中会提供的输入层张量。注意这是一个张量，字典里的内容是通过tf.estimator.inputs.numpy_input_fn中x参数的内容指定的。
#labels是正确答案，这个字段的内容是通过numpy_input_fn中y参数给出的。model的取值有3种可能，分别对应Estimator类的train、evaluate和predict这3个函数。通过这个参数可以判断当前是否是训练过程。
#最后params是一个字典，这个字典中可以给出模型相关的任何超参数（hyper-parameter）。比如这里将学习率放到params中。
def model_fn(features, labels, mode, params):
    #定义神经网络的结构并通过输入的到前向传播的结果
    predict = lenet(features["image"], mode == tf.estimator.ModeKeys.TRAIN)

    #如果在预测模式，那么只需要将结果返回即可
    if mode == tf.estimator.ModeKeys.PREDICT:
        #使用EstimatorSpec传递返回值，并通过predictions参数指定返回的结果
         return tf.estimator.EstimatorSpec(mode=mode, predictions={"result" : tf.argmax(predict, 1)})

    #定义损失函数
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=labels))
    #定义优化函数和训练过程
    train_op = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"]).minimize(loss=loss, global_step=tf.train.get_global_step())
    #定义评测标准，在运行evaluate时会计算这里定义的所有评测标准
    eval_metric_ops = {"my_metric": tf.metrics.accuracy(tf.argmax(predict, 1), labels)}
    #返回模型训练过程需要使用的损失函数、训练过程和评测方法
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

mnist = input_data.read_data_sets("E:\\Alls\\软件\\tensorflow-mnist", one_hot=False)

#通过自定义的方式生成Estimator类。这里需要提供模型定义的函数并通过params参数指定模型定义时使用的超参数
model_params = {"learning_rate": 0.01}
estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
#定义训练和评测模型
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"image": mnist.train.images}, y=mnist.train.labels.astype(np.int32), num_epochs=None, batch_size=128, shuffle=True)
estimator.train(input_fn=train_input_fn, steps=30000)
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"image": mnist.test.images}, y=mnist.test.labels.astype(np.int32), num_epochs=1, batch_size=128, shuffle=False)
test_result = estimator.evaluate(input_fn=test_input_fn)
#这里使用的my_metric中的内容就是model_fn中eval_metric_ops定义的评测指标
accuracy_score = test_result["my_metric"]
print("Test accuracy: %g %%" % (accuracy_score * 100))

#使用训练好的模型在新数据上预测结果
predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"image": mnist.test.images[:10]}, num_epochs=1, shuffle=False)
predictions = estimator.predict(input_fn=predict_input_fn)
for i, p in enumerate(predictions):
    print("Predictions %s: %s" % (i + 1, p["result"]))

目前比较主流的Tensorflow高层封装主要有4个，分别是Tensorflow-Slim、TFLearn、Keras和Estimator。
1.Tensorflow-Slim是Google官方给出的相对较早的Tensorflow高层封装，Google通过Tensorflow-Slim开源了一些已经训练好的图像分析的模型，所以目前在图像识别问题中Tensorflow-Slim仍被较多地使用。
更多参考https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/和https://github.com/tensorflow/models/tree/master/research/slim/nets

2.先使用pip install tflearn安装TFLearn，TFLearn训练流程也是先定义神经网络的结构，在使用训练数据来训练模型，与原生态Tensorflow不同的地方在于，TFLearn不仅使神经网络结构定义更加简洁，还对模型训练的过程也进行了封装。
另外，在定义神经网络的前向传播过程之后，TFLearn可以通过regression函数来指定损失函数和优化方法，更方便的是，不仅TFLearn能更好地封装模型定义，tflearn.DNN也能很好地封装模型训练的过程。
通过fit函数可以指定训练中使用的数据和训练的轮数。这样避免了大量的冗余代码。更多使用可以参考:http://tflearn.org/

3.keras
Keras是目前使用最为广泛的深度学习工具之一，它的底层可以支持Tensorflow、MXNet、CNTK和Theano。
3.1 Keras基本用法
和TFLearn API类似，Keras API也对模型定义、损失函数、训练过程等进行了封装，而且封装之后的整个训练过程和TFLearn是基本一致的，可以分为数据处理、模型定义和模型训练三个部分。
使用原生态的Keras API需要先安装Keras包，安装方式如下:pip install keras。更多可以参考官网：https://keras.io/

3.2 Keras高级用法
基础用法中最重要的封装就是Sequential类，所有的神经网络模型定义和训练都是通过Sequential实例来实现的。然而，从这个类的名称可以看出，它只支持顺序模型的定义。类似Inception这样的模型结构，
通过Sequential类就不容易直接实现了。为了支持更加灵活的模型定义方法，Keras支持以返回值的形式定义网络层结构。

3.3 Tensorflow集成Keras
虽然通过返回值的方式已经可以实现大部分的神经网络模型，然而Keras API还存在两大问题。第一，原生态Keras API对训练数据的处理流程支持得不太好，基本上需要一次性将数据全部加载到内存。
第二，原生态Keras API无法支持分布式训练，为了解决这两个问题，Keras提供了一种与原生态Tensorflow结合得更加紧密的方式。示例Keras_High_TF_K.py显示了如何将Keras和原生态Tensorflow API联合起来解决MNIST问题。

4.Estimator
Estimator是Tensorflow官方提供的高层API，所以它更好地整合了原生态Tensorflow提供的功能。
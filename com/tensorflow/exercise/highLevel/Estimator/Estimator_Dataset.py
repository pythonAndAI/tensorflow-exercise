import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

#实现Tensorflow和Dataset数据集结合的方式完成整个数据读取和模型训练的过程。
#通过下面代码可以看出Estimator可以非常好地和数据集结合，这样就能够很容易地支持海量数据读入或者复杂的数据预处理流程。

#Estimator的自定义输入函数需要每一次被调用时可以得到一个batch的数据（包括所有的输入层数据和期待的正确答案标注），通过数据集可以很自然地实现这个过程。
#虽然Estimator要求的自定义输入函数不能有参数，但是通过python提供的lambda表达式可以快速将下面的函数转化为不带参数的函数
def my_input_fn(filepath, perform_shuffle=False, repeat_count=1):
    #定义解析csv文件中一行的方法
    def decode_csv(line):
        #将一行中的数据解析出来，注意iris数据中最后一列为正确答案，前面4列为特征
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        #Estimator的输入函数要求特征是一个字典，所以这里返回的也需要是一个字典。字典中key的定义需要和DNNClassifier中ferture_columns的定义匹配。
        return {"x": parsed_line[:-1]}, parsed_line[-1:]

    #使用数据集处理输入数据。
    dataset = tf.data.TextLineDataset(filepath).skip(1).map(decode_csv)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count).batch(32)
    iterator = dataset.make_one_shot_iterator()
    #通过定义的数据集得到一个batch的输入数据。这个就是整个自定义的输入过程的返回结果
    batch_features, batch_labels = iterator.get_next()
    #如果为预测过程提供输入数据，那么batch_labels可以直接使用None。
    return batch_features, batch_labels

#定义Estimator
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 10], n_classes=3)

#使用lambda表达式将训练相关的信息传入自定义输入数据处理函数并生成Estimator需要的输入函数
classifier.train(input_fn=lambda: my_input_fn("", True, 10))

#使用lambda表达式将测试相关的信息传入自定义输入数据处理函数并生成Estimator需要的输入函数。通过lambda表达式的方式可以大大减少冗余的代码
test_results = classifier.evaluate(input_fn=lambda: my_input_fn("", False, 1))
print("Test accuracy: %g %%" % (test_results["accuracy"] * 100))
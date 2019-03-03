import tensorflow as tf

INPUT_NODES = 784
HIDDEN_NODES = 500
OUTPUT_NODES = 10

def get_weigth_variable(shape, regularizer):
    weigths = tf.get_variable(shape=shape, name="weigths", initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weigths))

    return weigths

#定义前向传播函数
def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weigths = get_weigth_variable([INPUT_NODES, HIDDEN_NODES], regularizer)
        biases = tf.get_variable("biases", [HIDDEN_NODES], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weigths) + biases)

    with tf.variable_scope("layer2"):
        weigths = get_weigth_variable([HIDDEN_NODES, OUTPUT_NODES], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODES], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weigths) + biases
    #返回最后的前向传播结果
    return layer2

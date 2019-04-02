import numpy as np

#用numpy实现一个简单的RNN前向传播过程
if __name__ == "__main__":
    X = [1, 2]
    state = [0, 0]
    #分开定义不同输入部分的权重以方便操作
    #上一时刻的状态的权重
    w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    #输入层权重
    w_cell_input = np.asarray([0.5, 0.6])
    #循环体偏置
    b_cell = np.asarray([0.1, -0.1])
    #定义用于输出的全连接层参数
    #权重
    w_output = np.asarray([1.0, 2.0])
    #偏置
    b_output = 0.1
    for i in range(len(X)):
        #计算循环体中的全连接层神经网络
        before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
        state = np.tanh(before_activation)
        #根据当前时刻状态计算最终输出
        final_output = np.dot(state, w_output) + b_output
        print("before activation:", before_activation)
        print("state" + str(i) + ":", state)
        print("final output:", final_output)

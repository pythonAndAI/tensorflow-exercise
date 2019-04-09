from keras.layers import Conv2D, MaxPooling2D, Input
import keras

#通过Keras实现类似Inception这样的模型结构。Inceptions的结构可以参考（../CNN/Inception-v3.png）路径的图片

#定义输入图像尺寸
input_img = Input(shape=(256, 256, 3))

#定义第一个分支
tower1 = Conv2D(64, (1, 1), padding="same", activation="relu")(input_img)
tower1 = Conv2D(64, (3, 3), padding="same", activation="relu")(tower1)

#定义第二个分支。与顺序模型不同，第二个分支的输入使用的是input_img，而不是第一个分支的输出
tower2 = Conv2D(64, (1, 1), padding="same", activation="relu")(input_img)
tower2 = Conv2D(64, (5, 5), padding="same", activation="relu")(tower2)

#定义第三个分支，类似的，第三个分支的输入也是input_img
tower3 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(input_img)
tower3 = Conv2D(64, (1, 1), padding="same", activation="relu")(tower3)

#将三个分支通过concatenate的方式拼接在一起
output = keras.layers.concatenate([tower1, tower2, tower3], axis=1)

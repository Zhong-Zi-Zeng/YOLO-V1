import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

class Network(Model):
    def __init__(self, S=7, B=2, C=20):
        """
        :param S: 需要將每張圖像分成幾等分
        :param B: 每個ceil需要預測多少個邊界框
        :param C: 預測類別數量
        """
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.output_channel = self.B * 5 + self.C  # B * 5 + C
        self.output_layer = OutputLayer(S, B, C)
        self.network = Sequential()

    def build_network(self):
        """
        :return: network , output shape (Batch, S, S, Output_channel), (c1, c2, x1, y1, w1, h1, x2, y2, w2, h2, p1, p2, ....)
        """
        self.network.add(Conv2D(64, (7, 7), (2, 2), padding='same', input_shape=(448, 448, 3), activation=tf.nn.leaky_relu))
        self.network.add(MaxPooling2D((2, 2), padding='same'))
        self.network.add(Conv2D(192, (3, 3), padding='same', activation=tf.nn.leaky_relu))
        self.network.add(MaxPooling2D((2, 2), padding='same'))
        self.network.add(Conv2D(128, (1, 1), padding='same', activation=tf.nn.leaky_relu))
        self.network.add(Conv2D(256, (3, 3), padding='same', activation=tf.nn.leaky_relu))
        self.network.add(Conv2D(256, (1, 1), padding='same', activation=tf.nn.leaky_relu))
        self.network.add(Conv2D(512, (3, 3), padding='same', activation=tf.nn.leaky_relu))
        self.network.add(MaxPooling2D((2, 2), padding='same'))

        for i in range(4):
            self.network.add(Conv2D(256, (1, 1), padding='same', activation=tf.nn.leaky_relu))
            self.network.add(Conv2D(512, (3, 3), padding='same', activation=tf.nn.leaky_relu))

        self.network.add(Conv2D(512, (1, 1), padding='same', activation=tf.nn.leaky_relu))
        self.network.add(Conv2D(1024, (3, 3), padding='same', activation=tf.nn.leaky_relu))
        self.network.add(MaxPooling2D((2, 2), padding='same'))

        for i in range(2):
            self.network.add(Conv2D(512, (1, 1), padding='same', activation=tf.nn.leaky_relu))
            self.network.add(Conv2D(1024, (3, 3), padding='same', activation=tf.nn.leaky_relu))

        self.network.add(Conv2D(1024, (3, 3), padding='same', activation=tf.nn.leaky_relu))
        self.network.add(Conv2D(1024, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.leaky_relu))
        self.network.add(Conv2D(1024, (3, 3), padding='same', activation=tf.nn.leaky_relu))
        self.network.add(Conv2D(1024, (3, 3), padding='same', activation=tf.nn.leaky_relu))
        self.network.add(Flatten())
        self.network.add(Dense(self.S * self.S * self.output_channel, activation=tf.nn.leaky_relu))
        self.network.add(self.output_layer)

        return self.network


class OutputLayer(Model):
    def __init__(self, S, B, C):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: (Batch, S * S * (B * 5 + C))
        :return:
        """

        # Reshape input to (Batch, S, S, B * 5 + C)
        inputs = tf.reshape(inputs, (-1, self.S, self.S, self.B * 5 + self.C))

        # pre bbox (Batch, S, S, 8)
        pre_bbox = inputs[:, :, :, 0:self.B * 4]
        pre_bbox = tf.nn.sigmoid(pre_bbox)

        # confidence (Batch, S, S, 2)
        conf = inputs[:, :, :, self.B * 4: self.B * 4 + self.B]
        conf = tf.nn.sigmoid(conf)

        # class prob (Batch, S, S, 20)
        class_prob = inputs[:, :, :,  self.B * 5::]
        class_prob = tf.nn.softmax(class_prob)

        output = tf.concat([pre_bbox, conf, class_prob], axis=3)

        return output

# model = Network().build_network()
#
# model.build(input_shape=(448, 448, 3))
# test = np.zeros((4, 448, 448, 3))
#
# model.compile()
# model.summary()
# test = tf.convert_to_tensor(test)
# output = model.predict(test)
# print(output.shape)

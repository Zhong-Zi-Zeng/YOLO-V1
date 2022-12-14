import tensorflow as tf
from Tools import bbox_coord_transform
from Network import Network
import cv2
import numpy as np
import os


TEST_IMAGE_PATH = "./VOC2007/JPEGImages"

test_img = cv2.imread(TEST_IMAGE_PATH + "/" + "000017.jpg")
test_img = cv2.resize(test_img, (448, 448), interpolation=cv2.INTER_AREA)
tensor_img = tf.convert_to_tensor(test_img / 255., dtype=tf.float64)


model = Network().build_network()
model.load_weights("./20221213.h5")


pred_y = model(test_img)
print(pred_y)
from Tools import transform_coord, calculate_iou, bbox_coord_transform
from Network import Network
import tensorflow as tf
from tensorflow.keras.losses import MSE
import cv2
import copy
import numpy as np


def bbox_loss(pre_bbox, label_bbox, S, B, img):
    """
    @brief:
        pre_bbox的x、y 為相對於ceil的座標, w、h為整張圖片0-1的範圍
        label_bbox的x, y, w, h為相對於整張圖片原始位置

    :param pre_bbox: (x1, y1, w1, h1, x2, y2, w2, h2), shape=(Batch, S, S, B * 4)
    :param label_bbox: (x, y, w, h), shape=(Batch, Box_number, 4)
    :param S 每張圖像分成幾等份
    :param B 每個ceil預測多少個邊界框
    :return:
    """
    # 將型態轉換為numpy
    if not isinstance(pre_bbox, np.ndarray):
        pre_bbox = pre_bbox.numpy()

    if not isinstance(label_bbox, np.ndarray):
        label_bbox = label_bbox.numpy()

    # 將預測方框的座標位置轉換成相對於整體圖片的座標
    pre_bbox = transform_coord(pre_bbox, S)

    # Reshape pre_bbox to (Batch, S, S, B, 4)
    pre_bbox = np.reshape(pre_bbox, ((-1, S, S, B, 4)))

    # Draw image
    # for bbox in pre_bbox.reshape((-1, 4)):
    #     copy_img = copy.copy(img)
    #     x1, y1, x2, y2 = bbox
    #     cv2.rectangle(copy_img, [int(x1), int(y1)], [int(x2), int(y2)], (255, 0, 0), 4)
    #     cv2.imshow("", copy_img)
    #     cv2.waitKey(0)

    # 紀錄損失
    total_bbox_loss = 0

    for b in range(label_bbox.shape[0]):
        for bbox_id in range(label_bbox.shape[1]):
            # 計算label_bbox對應到的是哪一個ceil
            ceil_row = label_bbox[b, bbox_id, 0] // (448 // S)
            ceil_col = label_bbox[b, bbox_id, 1] // (448 // S)

            # 找出與label擁有最大iou的預測方框，負責計算loss
            iou_list = calculate_iou(pre_bbox[b, ceil_row, ceil_col, :, :], label_bbox[b, bbox_id, :])
            max_iou_bbox = np.argmax(iou_list)
            responsible_bbox = pre_bbox[b, ceil_row, ceil_col, max_iou_bbox, :]

            responsible_bbox = tf.convert_to_tensor(responsible_bbox)
            label_bbox = tf.convert_to_tensor(label_bbox)

            # x, y座標回歸損失
            x_y_loss = 5 * (responsible_bbox[0] / 448 - label_bbox[b, bbox_id, 0] / 448) ** 2 + \
                       (responsible_bbox[1] / 448 - label_bbox[b, bbox_id, 1] / 448) ** 2


            # w、 h損失
            w_h_loss = 5 * (tf.sqrt(responsible_bbox[2] / 448) - tf.sqrt(np.float64(label_bbox[b, bbox_id, 2]) / 448)) +\
                       (tf.sqrt(responsible_bbox[3] / 448) - tf.sqrt(np.float64(label_bbox[b, bbox_id, 3])) / 448 ) ** 2

            total_bbox_loss += (x_y_loss + w_h_loss)
            print(total_bbox_loss)

    return total_bbox_loss



# pre_bbox = np.random.random(size=(1, 392)).reshape(1, 7, 7, 8)
# pre_bbox = np.abs(pre_bbox)

# pre_bbox = tf.fill((1, 7, 7, 8), 0.1)

# test = tf.random.normal((1, 2, 4))

# print(test)
# pre_bbox[:, :, :, 2::4] = tf.multiply(pre_bbox[:, :, :, 2::4], 448)
# print(pre_bbox)
# # print(test[:, :, 1:4:2])
# print(tf.gather(test, [0, 1], axis=0))


label_bbox = np.array([[[222, 194, 251, 224]]])
img = cv2.imread("Cat.png")
img = np.float64(img)
tensor_img = tf.convert_to_tensor(img[np.newaxis, :] / 255.)

# bbox_loss(pre_bbox, label_bbox, 7, 2, img)

model = Network().build_network()
model.compile( run_eagerly=True)

for e in range(1):
    with tf.GradientTape() as tape:
        output = model(tensor_img)
        output = output.numpy()
        pre_bbox = output[:, :, :, 0:8]
        total_bbox_loss = bbox_loss(pre_bbox, label_bbox, 7, 2, img)
        # total_bbox_loss = tf.convert_to_tensor(total_bbox_loss)

    grad = tape.gradient(total_bbox_loss, model.trainable_variables)
    print(grad)



# x = list(range(0, 449, 64))
# new_label_bbox = bbox_coord_transform(label_bbox[0])
# x1, y1, x2, y2 = new_label_bbox[0]
#
# for i in x:
#     cv2.line(img, (i, 0), (i, 448), (0, 0, 255), 2)
#     cv2.line(img, (0, i), (448, i), (0, 0, 255), 2)
#
# cv2.rectangle(img, [int(x1), int(y1)], [int(x2), int(y2)], (255, 0, 0), 4)
# cv2.imshow("", img)
# cv2.waitKey(0)



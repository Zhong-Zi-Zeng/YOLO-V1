from Tools import transform_coord, calculate_iou, bbox_coord_transform
from Network import Network
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import cv2
import copy
import numpy as np


def draw_image(global_pre_bbox, img):
    for bbox in tf.reshape(global_pre_bbox, (1 * 7 * 7 * 2, 4)):
        copy_img = copy.copy(img)
        x, y, w, h = bbox
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        print(x1, y1, x2, y2)

        cv2.rectangle(copy_img, [int(x1), int(y1)], [int(x2), int(y2)], (255, 0, 0), 4)
        cv2.imshow("", copy_img)
        cv2.waitKey(0)


def get_loss(pred_y, label_bbox, cls_label, S, B, C, img):
    """
    @brief:
        pre_bbox的x、y 為相對於ceil的座標, w、h為整張圖片0-1的範圍
        label_bbox的x, y, w, h為相對於整張圖片原始位置

    :param pred_y: (x1, y1, w1, h1, x2, y2, w2, h2, c1, c2, p1...), shape=(Batch, S, S, B * 5 + C)
    :param label_bbox: (x, y, w, h), shape=(Batch, S, S, 1, 4)
    :param cls_label: (1 , 0, 0...), shape=(Batch ,S, S, 20)
    :param S 每張圖像分成幾等份
    :param B 每個ceil預測多少個邊界框
    :return:
    """

    # 取出各輸出值
    pre_bbox = pred_y[..., 0:B * 4]  # (Batch, S, S, B * 4)
    pre_conf = pred_y[..., B * 4: B * 4 + B]  # (Batch, S, S, B)
    pre_cls_prob = pred_y[..., B * 5:]  # (Batch, S, S, C)

    # Reshape pre_bbox to (Batch, S, S, B, 4)
    pre_bbox = tf.reshape(pre_bbox, ((-1, S, S, B, 4)))

    # 將預測方框的座標位置轉換成相對於整體圖片的座標 (Batch, S, S, B, 4)
    global_pre_bbox = transform_coord(pre_bbox, S, B)

    # Draw image
    # draw_image(global_pre_bbox, img)

    # 計算iou (Batch, S, S, B)
    iou_score = calculate_iou(global_pre_bbox, label_bbox)

    # 取出有遮罩，在後面算損失的時候乘上遮罩即可過濾出負責預測邊界框的損失
    mask = tf.reduce_max(iou_score, axis=-1, keepdims=True)
    mask = tf.cast((iou_score == mask), tf.float32)  # 只包含0和1

    # 找出有物件的ceil (Batch, S, S, 1)
    label_response = tf.cast((label_bbox[..., 0] != 0), dtype=tf.float32)

    # 將剛剛的遮罩乘上label_response，則只會剩下負責預測的邊界框的mask
    obj_mask = mask * label_response  # (Batch, S, S, B)只包含0和1

    global_pre_bbox = tf.cast(global_pre_bbox, dtype=tf.float32)
    label_bbox = tf.cast(label_bbox, dtype=tf.float32)

    # 計算bbox損失
    xy_loss = 5 * (tf.square((global_pre_bbox[..., 0] - label_bbox[..., 0]) / 448) +
                   tf.square((global_pre_bbox[..., 1] - label_bbox[..., 1]) / 448))

    wh_loss = 5 * (tf.square((tf.sqrt(global_pre_bbox[..., 2]) - tf.sqrt(label_bbox[..., 2])) / 448) +
                   tf.square((tf.sqrt(global_pre_bbox[..., 3]) - tf.sqrt(label_bbox[..., 3])) / 448))

    bbox_loss = obj_mask * (xy_loss + wh_loss)
    bbox_loss = tf.reduce_sum(bbox_loss)

    # 計算pre_conf損失
    obj_loss = tf.square(1 - pre_conf)
    obj_loss = obj_mask * obj_loss

    non_obj_loss = tf.square(pre_conf)
    non_obj_loss = 0.5 * (1 - obj_mask) * non_obj_loss

    conf_loss = tf.reduce_sum(non_obj_loss + obj_loss)

    # 計算類別損失
    cls_loss = tf.square(pre_cls_prob - cls_label)
    cls_loss = cls_loss * label_response
    cls_loss = tf.reduce_sum(cls_loss)


    total_loss = bbox_loss + conf_loss + cls_loss
    print(total_loss)
    return total_loss


# pre_bbox = np.random.random(size=(1, 392)).reshape(1, 7, 7, 8)
# pre_bbox = np.abs(pre_bbox)

# pre_bbox = tf.random.uniform((1, 7, 7, 8))

# test = tf.random.normal((1, 2, 4))

# print(test)
# pre_bbox[:, :, :, 2::4] = tf.multiply(pre_bbox[:, :, :, 2::4], 448)
# print(pre_bbox)
# # print(test[:, :, 1:4:2])
# print(tf.gather(test, [0, 1], axis=0))

cls_label = np.zeros((1, 7, 7, 20))
cls_label[0, 3, 3, 0] = 1
cls_label[0, 3, 5, 2] = 1


label_bbox = np.zeros((1, 7, 7, 1, 4))
label_bbox[0, 3, 3] = [222, 194, 251, 224]
label_bbox[0, 3, 5] = [208, 366, 109, 77]

tf.random.set_seed(2)
img = cv2.imread("Cat.png")
img = np.float64(img)
tensor_img = tf.convert_to_tensor(img[np.newaxis, :] / 255.)

# bbox_loss(pre_bbox, label_bbox, 7, 2, img)

model = Network().build_network()
model.compile()
adam = Adam(learning_rate=0.01)

for i in range(20):
    with tf.GradientTape() as tape:
        output = model(tensor_img)
        total_bbox_loss = get_loss(output, label_bbox, cls_label, 7, 2, 20, img)

    grad = tape.gradient(total_bbox_loss, model.trainable_variables)
    adam.apply_gradients(zip(grad, model.trainable_variables))

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

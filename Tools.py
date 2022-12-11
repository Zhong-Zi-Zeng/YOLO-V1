import tensorflow as tf
import numpy as np

def transform_coord(pre_bbox, S, B):
    """
    @brief:
        將預測方框的座標位置轉換成相對於整體圖片的座標
        pre_bbox的x、y 為相對於ceil的座標, w、h為整張圖片0-1的範圍

    :param pre_bbox: (x, y, w, h), shape=(Batch, S, S, B, 4)
    :return: 轉換到全局座標的 pre_bbox: (x, y, w, h), shape=(Batch, S, S, B, 4)
    :param S 每張圖像分成幾等份
    """
    batch = pre_bbox.shape[0]

    # 轉換w、h
    pre_bbox_w = tf.reshape(pre_bbox[..., 2] * 448, (-1, S, S, B, 1))
    pre_bbox_h = tf.reshape(pre_bbox[..., 3] * 448, (-1, S, S, B, 1))

    # 轉換x、y
    offset_x = tf.reshape(tf.transpose(tf.tile(np.arange(S).reshape(1, S), multiples=[B, S * batch])), (batch, S, S, B, 1))
    offset_y = tf.reshape(tf.tile(np.arange(S).reshape(S, 1), multiples=[batch, S * B]), (batch, S, S, B, 1))

    offset_x = tf.cast(offset_x, dtype=tf.float32)
    offset_y = tf.cast(offset_y, dtype=tf.float32)

    pre_bbox_x = tf.reshape(pre_bbox[..., 0], (-1, S, S, B, 1)) + (offset_x / S * 448)
    pre_bbox_y = tf.reshape(pre_bbox[..., 1], (-1, S, S, B, 1)) + (offset_y / S * 448)

    return tf.concat([pre_bbox_x, pre_bbox_y, pre_bbox_w, pre_bbox_h], axis=4)

def bbox_coord_transform(bbox):
    """
    @brief:
        將bbox座標轉換為(x1, y1, x2, y2), shape = (Batch, S, S, B, 4)
    :param bbox: (m_x, m_y, w, h)
    :return: bbox(x1, y1, x2, y2)
    """

    new_bbox_coord = np.zeros_like(bbox)

    # x1
    new_bbox_coord[..., 0] = bbox[..., 0] - (bbox[..., 2] / 2)

    # y1
    new_bbox_coord[..., 1] = bbox[..., 1] - (bbox[..., 3] / 2)

    # x2
    new_bbox_coord[..., 2] = bbox[..., 0] + (bbox[..., 2] / 2)

    # y2
    new_bbox_coord[..., 3] = bbox[..., 1] + (bbox[..., 3] / 2)

    return new_bbox_coord


def calculate_iou(global_pre_bbox, label_bbox):
    """
    @brief:
        用來找出與目標方框之間的iou
        global_pre_bbox, y, w, h為相對於整張圖片原始位置
        label_bbox的x, y, w, h為相對於整張圖片原始位置

    :param global_pre_bbox: shape=(Batch, S, S, B, 4)
    :param label_bbox: (x, y, w, h), shape=(Batch, S, S, 1, 4)
    :return: 與目標方框有最大iou值的預測方框索引值
    """

    # 將座標轉換成x1, y1, x2, y2 (Batch, S, S, B, 4)
    global_pre_bbox = bbox_coord_transform(global_pre_bbox)
    label_bbox = bbox_coord_transform(label_bbox)

    # 計算label自己的面積 (1, S, S, 1)
    label_area = (label_bbox[..., 2] - label_bbox[..., 0]) * (label_bbox[..., 3] - label_bbox[..., 1])

    # 計算global_pre_bbox 自己的面積
    global_pre_bbox_area = (global_pre_bbox[..., 2] - global_pre_bbox[..., 0]) * (global_pre_bbox[..., 3] - global_pre_bbox[..., 1])

    # 找出重疊區域面積 (1, S, S, B)
    intersect_w = tf.minimum(global_pre_bbox[..., 2], label_bbox[..., 2]) - tf.maximum(global_pre_bbox[..., 0], label_bbox[..., 0])
    intersect_h = tf.minimum(global_pre_bbox[..., 3], label_bbox[..., 3]) - tf.maximum(global_pre_bbox[..., 1], label_bbox[..., 1])
    intersect_area = intersect_w * intersect_h
    intersect_area = tf.maximum(intersect_area, 0)

    # (1, S, S, B)
    iou_score = intersect_area / (label_area + intersect_area + global_pre_bbox_area)

    return iou_score

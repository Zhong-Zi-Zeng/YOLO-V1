import tensorflow as tf
import numpy as np

def transform_coord(pre_bbox, S):
    """
    @brief:
        將預測方框的座標位置轉換成相對於整體圖片的座標
        pre_bbox的x、y 為相對於ceil的座標, w、h為整張圖片0-1的範圍

    :param pre_bbox: (x1, y1, w1, h1, x2, y2, w2, h2), shape=(Batch, S, S, B * 4)
    :return: 轉換到全局座標的pre_bbox
    :param S 每張圖像分成幾等份
    """

    # 轉換w、h
    pre_bbox[:, :, :, 2::4] *= 448
    pre_bbox[:, :, :, 3::4] *= 448

    # 轉換x、y
    offset_x = np.tile(np.arange(7), reps=[7]).reshape((1, S, S, 1))
    offset_y = np.tile(np.arange(7).reshape((1, 7)), reps=[7, 1]).transpose().reshape((1, S, S, 1))

    pre_bbox[:, :, :, 0::4] += (offset_x / S * 448)
    pre_bbox[:, :, :, 1::4] += (offset_y / S * 448)

    return pre_bbox


def bbox_coord_transform(bbox):
    """
    @brief:
        將bbox座標轉換為(x1, y1, x2, y2), shape = (bbox_number, 4)
    :param bbox: (m_x, m_y, w, h)
    :return: bbox(x1, y1, x2, y2)
    """

    if not isinstance(bbox, np.ndarray):
        bbox = bbox.numpy()

    new_bbox_coord = np.zeros_like(bbox)

    # x1
    new_bbox_coord[:, 0] = bbox[:, 0] - (bbox[:, 2] / 2)

    # y1
    new_bbox_coord[:, 1] = bbox[:, 1] - (bbox[:, 3] / 2)

    # x2
    new_bbox_coord[:, 2] = bbox[:, 0] + (bbox[:, 2] / 2)

    # y2
    new_bbox_coord[:, 3] = bbox[:, 1] + (bbox[:, 3] / 2)

    return new_bbox_coord


def calculate_iou(pre_bbox, label_bbox):
    """
    @brief:
        用來找出與目標方框之間的iou
        pre_bbox的x, y, w, h為相對於整張圖片原始位置
        label_bbox的x, y, w, h為相對於整張圖片原始位置

    :param pre_bbox: (pre_bbox_number, x, y, w, h), shape=(pre_bbox_number, 4)
    :param label_bbox: (x, y, w, h)
    :return: 與目標方框有最大iou值的預測方框索引值
    """

    # 將座標轉換成x1, y1, x2, y2
    pre_bbox = bbox_coord_transform(pre_bbox)
    label_bbox = bbox_coord_transform(label_bbox[np.newaxis, :])[0]

    # 計算label自己的面積
    label_area = (label_bbox[2] - label_bbox[0]) * (label_bbox[3] - label_bbox[1])

    # 紀錄iou
    iou_history = []

    for bbox in pre_bbox:
        # 預測方框的面積
        pre_bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # 找出重疊區域的面積
        w = np.min([bbox[2], label_bbox[2]]) - np.max([bbox[0], label_bbox[0]])
        h = np.min([bbox[3], label_bbox[3]]) - np.max([bbox[1], label_bbox[1]])

        if w <= 0 or h <= 0:
            iou_history.append(0)
            continue

        # 計算iou
        overlapping_area = w * h
        iou = overlapping_area / (label_area + pre_bbox_area - overlapping_area)
        iou_history.append(iou)

    return iou_history
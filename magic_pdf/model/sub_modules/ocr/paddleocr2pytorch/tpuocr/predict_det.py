import argparse
import logging
import os
import time

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from untool import EngineOV

logging.basicConfig(level=logging.DEBUG)


class PPOCRv2Det(object):
    def __init__(self, args):
        # load bmodel
        model_path = args.bmodel_det
        self.net = EngineOV(model_path, args.dev_id)

        # 预处理参数
        self.input_size = 640  # 固定输入尺寸
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        self.scale = (
            np.array([1 / 0.229, 1 / 0.224, 1 / 0.225], dtype=np.float32) / 255.0
        )

        # 后处理参数
        self.thresh = 0.3
        self.box_thresh = 0.6
        self.unclip_ratio = 1.5
        self.min_size = 3
        self.max_candidates = 1000

        # 统计时间
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, img):
        h, w = img.shape[:2]

        # 计算缩放比例，保证缩放后的尺寸不超过input_size
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 确保new_h和new_w都是正数且不超过input_size
        new_h = max(1, min(new_h, self.input_size))
        new_w = max(1, min(new_w, self.input_size))

        # 缩放图片
        if scale != 1.0:
            img = cv2.resize(img, (new_w, new_h))

        # 归一化
        img = img.astype(np.float32)
        img = (img - self.mean) * self.scale

        # 转换为CHW格式
        img = np.transpose(img, (2, 0, 1))

        # 填充到固定尺寸
        padded_img = np.zeros((3, self.input_size, self.input_size), dtype=np.float32)
        padded_img[:, :new_h, :new_w] = img

        return padded_img, scale, (h, w), (new_h, new_w)

    def predict(self, tensor):
        """模型推理"""
        outputs = self.net([np.array(tensor, dtype=np.float32)])[0]
        return outputs

    def unclip(self, box):
        try:
            poly = Polygon(box)
            distance = poly.area * self.unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = np.array(offset.Execute(distance))
            if len(expanded) > 0:
                return expanded[0]  # 取第一个结果
            else:
                return box
        except Exception as e:
            logging.warning(f"unclip操作失败: {e}")
            return box

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, box):
        """计算文本框的置信度分数"""
        h, w = bitmap.shape[:2]
        box = box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height, scale):
        height, width = bitmap.shape

        try:
            # 使用findContours找轮廓
            contours, _ = cv2.findContours(
                (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
        except Exception as e:
            logging.error(f"findContours失败: {e}")
            return []

        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []

        for index in range(num_contours):
            contour = contours[index]

            # 获取最小外接矩形
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            points = np.array(points)

            # 计算置信度分数
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if score < self.box_thresh:
                continue

            # 使用unclip扩展文本框
            expanded_points = self.unclip(points)
            expanded_points, sside = self.get_mini_boxes(
                expanded_points.reshape(-1, 1, 2)
            )
            if sside < self.min_size + 2:
                continue

            # 转换坐标到原图尺寸
            box = np.array(expanded_points)
            box[:, 0] = np.clip(box[:, 0] / scale, 0, dest_width - 1)
            box[:, 1] = np.clip(box[:, 1] / scale, 0, dest_height - 1)

            boxes.append(box.astype(np.int32))
            scores.append(score)

        # logging.info(f"检测到 {len(boxes)} 个候选框")
        return boxes

    def postprocess(self, pred, scale, orig_shape, resized_shape):
        """后处理，提取文本框"""
        orig_h, orig_w = orig_shape
        resized_h, resized_w = resized_shape

        try:
            # 获取有效区域的预测结果
            valid_h = min(resized_h, pred.shape[2])
            valid_w = min(resized_w, pred.shape[3])
            valid_pred = pred[0, 0, :valid_h, :valid_w]  # 注意输出是4维的

            # logging.info(f"有效预测区域: {valid_pred.shape}")
            # logging.info(
            #     f"预测值统计: min={valid_pred.min():.4f}, max={valid_pred.max():.4f}, mean={valid_pred.mean():.4f}"
            # )

            # 二值化
            bitmap = (valid_pred > self.thresh).astype(np.uint8)

            # 统计二值化结果
            positive_pixels = np.sum(bitmap > 0)
            # logging.info(f"二值化后正像素数量: {positive_pixels}/{bitmap.size}")

            if positive_pixels < 10:
                logging.warning("二值化后正像素太少，可能没有文本")
                return []

            # 使用boxes_from_bitmap提取文本框
            boxes = self.boxes_from_bitmap(valid_pred, bitmap, orig_w, orig_h, scale)

            return boxes

        except Exception as e:
            logging.error(f"后处理过程中发生错误: {e}")
            import traceback

            traceback.print_exc()
            return []

    def filter_tag_det_res(self, dt_boxes, image_shape):
        # 处理空列表的情况
        if len(dt_boxes) == 0:
            return np.array([])
            
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def __call__(self, img_list):
        img_num = len(img_list)
        dt_boxes_list = []

        # 对每张图片进行处理
        start_prep = time.time()
        for img in img_list:
            # 预处理
            input_data, scale, orig_shape, resized_shape = self.preprocess(img)
            input_batch = np.expand_dims(input_data, axis=0)

            # 推理
            start_infer = time.time()
            outputs = self.predict(input_batch)
            self.inference_time += time.time() - start_infer

            # 后处理
            start_post = time.time()
            boxes = self.postprocess(outputs, scale, orig_shape, resized_shape)

            # 过滤检测框
            if len(boxes) > 0:
                boxes = self.filter_tag_det_res(
                    boxes, (orig_shape[0], orig_shape[1], 3)
                )

            dt_boxes_list.append(boxes)
            self.postprocess_time += time.time() - start_post

        self.preprocess_time += time.time() - start_prep
        return dt_boxes_list

    def detect_single(self, img):
        """单图检测接口 - 直接处理单张图片"""
        # 预处理
        input_data, scale, orig_shape, resized_shape = self.preprocess(img)
        input_batch = np.expand_dims(input_data, axis=0)

        # 推理
        start_infer = time.time()
        outputs = self.predict(input_batch)
        self.inference_time += time.time() - start_infer

        # 后处理
        start_post = time.time()
        boxes = self.postprocess(outputs, scale, orig_shape, resized_shape)

        # 过滤检测框
        if len(boxes) > 0:
            boxes = self.filter_tag_det_res(boxes, (orig_shape[0], orig_shape[1], 3))
        else:
            # 确保返回空的numpy数组而不是空列表
            boxes = np.array([])

        self.postprocess_time += time.time() - start_post
        return boxes


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def main(opt):
    draw_img_save = "./results/det_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    ppocrv2_det = PPOCRv2Det(opt)
    # 读取得到的图片存放在这个list中
    file_list = sorted(os.listdir(opt.input))
    img_list = []
    for img_name in file_list:
        # label = img_name.split('.')[0]
        img_file = os.path.join(opt.input, img_name)
        # print(img_file, label)
        src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
        img_list.append(src_img)
    # 检测得到的结果
    dt_boxes_list = ppocrv2_det(img_list)

    for img_name, dt_boxes in zip(file_list, dt_boxes_list):
        image_file = os.path.join(opt.input, img_name)
        draw_im = draw_text_det_res(dt_boxes, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save, "det_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, draw_im)
        logging.info("The visualized image saved in {}".format(img_path))


def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument("--dev_id", type=int, default=0, help="tpu card id")
    parser.add_argument(
        "--input",
        type=str,
        default="../datasets/cali_set_det",
        help="input image directory path",
    )
    parser.add_argument(
        "--bmodel_det",
        type=str,
        default="../models/BM1684X/ch_PP-OCRv4_det_fp32.bmodel",
        help="bmodel path",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

import argparse
import copy
import os.path
import warnings

import cv2
import numpy as np
from loguru import logger

from magic_pdf.libs.config_reader import get_local_models_dir
from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.ocr_utils import (
    check_img,
    get_rotate_crop_image,
    merge_det_boxes,
    preprocess_image,
    sorted_boxes,
    update_det_boxes,
)

from . import predict_det, predict_rec

ocr_models_dir = os.path.join(get_local_models_dir(), "OCR")


class TPUTextSystem:
    def __init__(self, enable_long_image_slice=True, *args, **kwargs):
        # 创建args对象
        tpu_args = argparse.Namespace()
        tpu_args.dev_id = 0
        tpu_args.batch_size = 1
        tpu_args.bmodel_det = os.path.join(
            ocr_models_dir, "ch_PP-OCRv4_det_f16.bmodel"
        )
        tpu_args.bmodel_rec = os.path.join(
            ocr_models_dir, "ch_PP-OCRv4_rec_f16.bmodel"
        )
        tpu_args.char_dict_path = os.path.join(ocr_models_dir, "ppocr_keys_v1.txt")
        tpu_args.det_limit_side_len = [640]
        tpu_args.img_size = [[640, 48], [320, 48]]
        tpu_args.use_space_char = True
        tpu_args.use_beam_search = False
        tpu_args.beam_size = 5
        tpu_args.rec_thresh = 0.5

        # 初始化检测和识别模型
        self.text_detector = predict_det.PPOCRv2Det(tpu_args)
        self.text_recognizer = predict_rec.PPOCRv2Rec(tpu_args)
        self.drop_score = tpu_args.rec_thresh
        self.enable_long_image_slice = enable_long_image_slice

    def ocr(
        self,
        img,
        det=True,
        rec=True,
        mfd_res=None,
        tqdm_enable=False,
    ):
        """OCR接口，保持与原接口一致"""
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det:
            logger.error("When input a list of images, det must be false")
            exit(0)
        img = check_img(img)
        imgs = [img]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if det and rec:
                ocr_res = []
                for img in imgs:
                    img = preprocess_image(img)
                    dt_boxes, rec_res = self.__call__(img, mfd_res=mfd_res)
                    if not dt_boxes and not rec_res:
                        ocr_res.append(None)
                        continue
                    tmp_res = [
                        [box.tolist(), res] for box, res in zip(dt_boxes, rec_res)
                    ]
                    ocr_res.append(tmp_res)
                return ocr_res
            elif det and not rec:
                ocr_res = []
                for img in imgs:
                    img = preprocess_image(img)
                    dt_boxes = self.text_detector.detect_single(img)
                    if dt_boxes is None:
                        ocr_res.append(None)
                        continue
                    dt_boxes = sorted_boxes(dt_boxes)
                    dt_boxes = merge_det_boxes(dt_boxes)
                    if mfd_res:
                        dt_boxes = update_det_boxes(dt_boxes, mfd_res)
                    tmp_res = [box.tolist() for box in dt_boxes]
                    ocr_res.append(tmp_res)
                return ocr_res
            elif not det and rec:
                ocr_res = []
                for img in imgs:
                    if not isinstance(img, list):
                        img = preprocess_image(img)
                        img = [img]
                    rec_res = []
                    for single_img in img:
                        if self.enable_long_image_slice:
                            # 使用支持切片的识别方法
                            text, score = self.text_recognizer.recognize_single(single_img)
                        else:
                            # 使用原始识别方法
                            img_input = self.text_recognizer.preprocess(single_img)
                            if img_input is None:
                                text, score = ("", 0.0)
                            else:
                                width = img_input.shape[2]
                                stage = self.text_recognizer.get_stage_for_size_and_batch(width, 1)
                                if stage != self.text_recognizer.cur_stage:
                                    self.text_recognizer.net.reset_net_stage(0, stage)
                                    self.text_recognizer.cur_stage = stage
                                self.text_recognizer.input_shape = self.text_recognizer.stage_shapes[stage]
                                
                                img_batch = np.expand_dims(img_input, axis=0)
                                outputs = self.text_recognizer.predict(img_batch)
                                res = self.text_recognizer.postprocess(outputs, self.text_recognizer.beam_search, self.text_recognizer.beam_size)
                                
                                if res:
                                    text, score = res[0]
                                else:
                                    text, score = ("", 0.0)
                        rec_res.append((text, score))
                    ocr_res.append(rec_res)
                return ocr_res

    def __call__(self, img, mfd_res=None):
        """核心调用方法"""
        if img is None:
            logger.debug("no valid image provided")
            return None, None

        # 检测文本框
        dt_boxes = self.text_detector.detect_single(img)

        if dt_boxes is None or len(dt_boxes) == 0:
            logger.debug("no dt_boxes found")
            return None, None

        # 裁剪文本区域并识别
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        dt_boxes = merge_det_boxes(dt_boxes)

        if mfd_res:
            dt_boxes = update_det_boxes(dt_boxes, mfd_res)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)

        # 识别文本
        rec_res = []
        for img_crop in img_crop_list:
            if self.enable_long_image_slice:
                # 使用支持切片的识别方法
                text, score = self.text_recognizer.recognize_single(img_crop)
            else:
                # 使用原始识别方法（不支持切片）
                img_input = self.text_recognizer.preprocess(img_crop)
                if img_input is None:
                    text, score = ("", 0.0)
                else:
                    width = img_input.shape[2]
                    stage = self.text_recognizer.get_stage_for_size_and_batch(width, 1)
                    if stage != self.text_recognizer.cur_stage:
                        self.text_recognizer.net.reset_net_stage(0, stage)
                        self.text_recognizer.cur_stage = stage
                    self.text_recognizer.input_shape = self.text_recognizer.stage_shapes[stage]
                    
                    img_batch = np.expand_dims(img_input, axis=0)
                    outputs = self.text_recognizer.predict(img_batch)
                    res = self.text_recognizer.postprocess(outputs, self.text_recognizer.beam_search, self.text_recognizer.beam_size)
                    
                    if res:
                        text, score = res[0]
                    else:
                        text, score = ("", 0.0)
            
            rec_res.append((text, score))

        # 过滤低置信度结果
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res


if __name__ == "__main__":
    tpu_ocr = TPUTextSystem()
    img = cv2.imread("/data2/MinerU/demo/Snipaste_2025-06-25_14-29-49.png")
    dt_boxes, rec_res = tpu_ocr(img)
    ocr_res = []
    if not dt_boxes and not rec_res:
        ocr_res.append(None)
    else:
        tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        ocr_res.append(tmp_res)
    print(ocr_res)

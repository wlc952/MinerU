import argparse
import logging
import os
import time

import cv2
import numpy as np
from untool import EngineOV

logging.basicConfig(level=logging.DEBUG)


class PPOCRv2Rec(object):
    def __init__(self, args):
        # load bmodel
        model_path = args.bmodel_rec
        self.net = EngineOV(model_path, args.dev_id)
        self.cur_stage = 0  # 当前stage

        # 定义不同stage对应的输入形状
        self.stage_shapes = {
            0: [1, 3, 48, 320],  # stage 0
            1: [1, 3, 48, 640],  # stage 1
            2: [4, 3, 48, 320],  # stage 2
            3: [4, 3, 48, 640],  # stage 3
        }

        self.input_shape = [1, 3, 48, 640]  # 默认输入形状
        self.rec_batch_size = self.input_shape[0]  # Max batch size in model stages.
        self.img_size = args.img_size
        self.img_size = sorted(self.img_size, key=lambda x: x[0])
        self.img_ratio = [x[0] / x[1] for x in self.img_size]
        self.img_ratio = sorted(self.img_ratio)
        # 解析字符字典
        self.character = ["blank"]
        with open(args.char_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                self.character.append(line)
        if args.use_space_char:
            self.character.append(" ")
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.beam_search = args.use_beam_search
        self.beam_size = args.beam_size

    def preprocess(self, img):
        start_prep = time.time()
        h, w, _ = img.shape
        ratio = w / float(h)
        if ratio > self.img_ratio[-1]:
            resized_w = self.img_size[-1][0]
            resized_h = self.img_size[-1][1]
            padding_w = resized_w
        else:
            for max_ratio in self.img_ratio:
                if ratio <= max_ratio:
                    resized_h = self.img_size[0][1]
                    resized_w = int(resized_h * ratio)
                    padding_w = int(resized_h * max_ratio)
                    break

        if h != resized_h or w != resized_w:
            img = cv2.resize(img, (resized_w, resized_h))
        img = img.astype("float32")
        img = np.transpose(img, (2, 0, 1))
        img -= 127.5
        img *= 0.0078125

        padding_im = np.zeros((3, resized_h, padding_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = img

        self.preprocess_time += time.time() - start_prep
        return padding_im

    def slice_long_image(self, img, max_width=640, overlap=32):
        """
        对超长图片进行切片处理
        Args:
            img: 输入图像 (H, W, C)
            max_width: 最大宽度，超过此宽度的图片会被切片
            overlap: 切片之间的重叠像素数，防止文字被切断
        Returns:
            切片列表和对应的位置信息
        """
        h, w, c = img.shape
        
        # 如果图片宽度小于等于最大宽度，直接返回
        if w <= max_width:
            return [img], [(0, w)]
        
        # 计算切片参数
        step = max_width - overlap
        slices = []
        positions = []
        
        # 对图片进行切片
        for start_x in range(0, w, step):
            end_x = min(start_x + max_width, w)
            
            # 如果剩余宽度太小，扩展到包含足够的像素
            if end_x - start_x < max_width // 2 and len(slices) > 0:
                start_x = w - max_width
                end_x = w
            
            slice_img = img[:, start_x:end_x, :]
            slices.append(slice_img)
            positions.append((start_x, end_x))
            
            # 如果已经到达图片末尾，跳出循环
            if end_x >= w:
                break
                
        return slices, positions

    def merge_slice_results(self, slice_results, positions, overlap=32):
        """
        合并切片识别结果
        Args:
            slice_results: 各个切片的识别结果列表 [(text, confidence), ...]
            positions: 各个切片的位置信息 [(start_x, end_x), ...]
            overlap: 重叠区域大小
        Returns:
            合并后的文本和置信度
        """
        if not slice_results:
            return ("", 0.0)
        
        if len(slice_results) == 1:
            return slice_results[0]
        
        # 合并文本
        merged_text = ""
        total_confidence = 0.0
        valid_count = 0
        
        for i, (text, conf) in enumerate(slice_results):
            if text.strip():  # 只处理非空文本
                # 对于有重叠的切片，进行重复文本去除
                if i > 0 and overlap > 0:
                    # 简单的重复文本去除：如果当前文本的开头与前一个的结尾相似，则去除重复部分
                    prev_text = slice_results[i-1][0]
                    if prev_text and len(prev_text) > 2 and len(text) > 2:
                        # 查找可能的重复部分
                        overlap_chars = min(len(prev_text), len(text), overlap // 10)  # 估算重叠字符数
                        for j in range(overlap_chars, 0, -1):
                            if prev_text[-j:] == text[:j]:
                                text = text[j:]
                                break
                
                merged_text += text
                total_confidence += conf
                valid_count += 1
        
        # 计算平均置信度
        avg_confidence = total_confidence / valid_count if valid_count > 0 else 0.0
        
        return (merged_text, avg_confidence)

    def predict(self, tensor):
        start_infer = time.time()

        outputs = self.net([np.array(tensor, dtype=np.float32)])[0]
        self.inference_time += time.time() - start_infer
        return outputs

    def postprocess(self, outputs, beam_search=False, beam_width=5):
        start_post = time.time()
        result_list = []

        if beam_search:
            max_seq_len = outputs.shape[1]

            for batch_idx in range(outputs.shape[0]):
                beams = [{"prefix": [], "score": 1.0, "confs": []}]

                for t in range(max_seq_len):
                    new_beams = []

                    for beam in beams:
                        next_char_probs = outputs[batch_idx, t]
                        top_candidates = np.argsort(-next_char_probs)[:beam_width]

                        for c in top_candidates:
                            new_prefix = beam["prefix"] + [c]
                            new_score = beam["score"] * next_char_probs[c]
                            new_confs = beam["confs"] + [next_char_probs[c]]
                            new_beams.append(
                                {
                                    "prefix": new_prefix,
                                    "score": new_score,
                                    "confs": new_confs,
                                }
                            )

                    new_beams.sort(key=lambda x: -x["score"])
                    beams = new_beams[:beam_width]

                best_beam = max(beams, key=lambda x: x["score"])

                char_list = []
                conf_list = []
                pre_c = best_beam["prefix"][0]
                if pre_c != 0:
                    char_list.append(self.character[pre_c])
                    conf_list.append(best_beam["confs"][0])
                for idx, c in enumerate(best_beam["prefix"]):
                    if (pre_c == c) or (c == 0):
                        if c == 0:
                            pre_c = c
                        continue
                    char_list.append(self.character[c])
                    conf_list.append(best_beam["confs"][idx])
                    pre_c = c
                result_list.append(("".join(char_list), np.mean(conf_list)))

        else:  # original postprocess
            preds_idx = outputs.argmax(axis=2)
            preds_prob = outputs.max(axis=2)
            for batch_idx, pred_idx in enumerate(preds_idx):
                char_list = []
                conf_list = []
                pre_c = pred_idx[0]
                if pre_c != 0:
                    char_list.append(self.character[pre_c])
                    conf_list.append(preds_prob[batch_idx][0])
                for idx, c in enumerate(pred_idx):
                    if (pre_c == c) or (c == 0):
                        if c == 0:
                            pre_c = c
                        continue
                    char_list.append(self.character[c])
                    conf_list.append(preds_prob[batch_idx][idx])
                    pre_c = c

                result_list.append(("".join(char_list), np.mean(conf_list)))

        self.postprocess_time += time.time() - start_post
        return result_list

    def get_stage_for_size_and_batch(self, width, batch_size):
        """根据图像宽度和批次大小确定使用哪个stage"""
        if width <= 320 and batch_size == 1:
            return 0  # [1, 3, 48, 320]
        elif width <= 640 and batch_size == 1:
            return 1  # [1, 3, 48, 640]
        elif width <= 320 and batch_size <= 4:
            return 2  # [4, 3, 48, 320]
        elif width <= 640 and batch_size <= 4:
            return 3  # [4, 3, 48, 640]
        else:
            # 对于更大的尺寸，使用最大的stage
            return 3

    def __call__(self, img_list):
        img_dict = {}
        slice_info = {}  # 记录切片信息
        
        for img_size in self.img_size:
            img_dict[img_size[0]] = {"imgs": [], "ids": [], "res": []}
        
        for id, img in enumerate(img_list):
            h, w, _ = img.shape
            ratio = w / float(h)
            
            # 检查是否需要切片处理
            max_supported_ratio = self.img_ratio[-1]
            needs_slicing = ratio > max_supported_ratio * 1.5 or w > 800
            
            if needs_slicing:
                # 对超长图片进行切片
                slices, positions = self.slice_long_image(img, max_width=600, overlap=48)
                slice_info[id] = {"positions": positions, "slice_count": len(slices)}
                
                # 对每个切片进行预处理
                for slice_idx, slice_img in enumerate(slices):
                    processed_img = self.preprocess(slice_img)
                    if processed_img is not None:
                        img_dict[processed_img.shape[2]]["imgs"].append(processed_img)
                        # 使用复合ID来标记切片：原图ID + 切片索引
                        img_dict[processed_img.shape[2]]["ids"].append((id, slice_idx))
            else:
                # 正常预处理
                img = self.preprocess(img)
                if img is None:
                    continue
                img_dict[img.shape[2]]["imgs"].append(img)
                img_dict[img.shape[2]]["ids"].append(id)

        # 批处理推理
        for size_w in img_dict.keys():
            img_num = len(img_dict[size_w]["imgs"])

            if size_w > 640:
                # 对于大于640的宽度，单张处理
                stage = self.get_stage_for_size_and_batch(size_w, 1)
                self.net.reset_net_stage(0, stage)
                self.input_shape = self.stage_shapes[stage]
                self.rec_batch_size = self.input_shape[0]

                for img_input in img_dict[size_w]["imgs"]:
                    img_input = np.expand_dims(img_input, axis=0)
                    outputs = self.predict(img_input)
                    res = self.postprocess(outputs, self.beam_search, self.beam_size)
                    img_dict[size_w]["res"].extend(res)
            else:
                # 根据可能的批次大小选择合适的stage
                max_batch_size = min(4, img_num)  # 最大批次为4
                stage = self.get_stage_for_size_and_batch(size_w, max_batch_size)
                self.net.reset_net_stage(0, stage)
                self.input_shape = self.stage_shapes[stage]
                self.rec_batch_size = self.input_shape[0]

                for beg_img_no in range(0, img_num, self.rec_batch_size):
                    end_img_no = min(img_num, beg_img_no + self.rec_batch_size)
                    current_batch_size = end_img_no - beg_img_no

                    if current_batch_size < self.rec_batch_size:
                        # 如果当前批次小于模型批次大小，使用单张处理
                        stage = self.get_stage_for_size_and_batch(size_w, 1)
                        self.net.reset_net_stage(0, stage)
                        self.input_shape = self.stage_shapes[stage]

                        for ino in range(beg_img_no, end_img_no):
                            img_input = np.expand_dims(
                                img_dict[size_w]["imgs"][ino], axis=0
                            )
                            outputs = self.predict(img_input)
                            res = self.postprocess(
                                outputs, self.beam_search, self.beam_size
                            )
                            img_dict[size_w]["res"].extend(res)
                    else:
                        # 批量处理
                        img_input = np.stack(
                            img_dict[size_w]["imgs"][beg_img_no:end_img_no]
                        )
                        outputs = self.predict(img_input)
                        res = self.postprocess(
                            outputs, self.beam_search, self.beam_size
                        )
                        img_dict[size_w]["res"].extend(res)

        # 重组结果，合并切片结果
        rec_res = {"res": [], "ids": []}
        slice_results = {}  # 临时存储切片结果 {original_id: [(slice_idx, result), ...]}
        
        for size_w in img_dict.keys():
            for res, img_id in zip(img_dict[size_w]["res"], img_dict[size_w]["ids"]):
                if isinstance(img_id, tuple):  # 这是一个切片
                    original_id, slice_idx = img_id
                    if original_id not in slice_results:
                        slice_results[original_id] = []
                    slice_results[original_id].append((slice_idx, res))
                else:  # 正常图片
                    rec_res["res"].append(res)
                    rec_res["ids"].append(img_id)
        
        # 合并切片结果
        for original_id, slices in slice_results.items():
            # 按切片索引排序
            slices.sort(key=lambda x: x[0])
            slice_res_list = [res for _, res in slices]
            positions = slice_info[original_id]["positions"]
            
            # 合并切片结果
            merged_result = self.merge_slice_results(slice_res_list, positions, overlap=48)
            rec_res["res"].append(merged_result)
            rec_res["ids"].append(original_id)
        
        return rec_res

    def recognize_single(self, img):
        """单图识别接口 - 直接处理单张图片，支持超长图片切片"""
        h, w, _ = img.shape
        ratio = w / float(h)
        
        # 检查是否需要切片处理（宽高比过大或宽度过大）
        max_supported_ratio = self.img_ratio[-1]  # 通常是 640/48 = 13.33
        needs_slicing = ratio > max_supported_ratio * 2  # 如果比例太大
        
        if needs_slicing:
            # 对超长图片进行切片处理
            slices, positions = self.slice_long_image(img, max_width=600, overlap=48)
            slice_results = []
            
            for slice_img in slices:
                # 对每个切片进行预处理和识别
                img_input = self.preprocess(slice_img)
                if img_input is None:
                    slice_results.append(("", 0.0))
                    continue
                
                # 获取图像宽度以选择合适的stage
                width = img_input.shape[2]
                stage = self.get_stage_for_size_and_batch(width, 1)
                if stage != self.cur_stage:
                    self.net.reset_net_stage(0, stage)
                    self.cur_stage = stage
                self.input_shape = self.stage_shapes[stage]
                self.rec_batch_size = self.input_shape[0]
                
                # 单张图片推理
                img_batch = np.expand_dims(img_input, axis=0)
                outputs = self.predict(img_batch)
                res = self.postprocess(outputs, self.beam_search, self.beam_size)
                
                if res:
                    slice_results.append(res[0])
                else:
                    slice_results.append(("", 0.0))
            
            # 合并切片结果
            return self.merge_slice_results(slice_results, positions, overlap=48)
        
        else:
            # 正常处理流程
            img_input = self.preprocess(img)
            if img_input is None:
                return ("", 0.0)

            # 获取图像宽度以选择合适的stage
            width = img_input.shape[2]
            stage = self.get_stage_for_size_and_batch(width, 1)
            if stage != self.cur_stage:
                self.net.reset_net_stage(0, stage)
                self.cur_stage = stage
            self.input_shape = self.stage_shapes[stage]
            self.rec_batch_size = self.input_shape[0]

            # 单张图片推理
            img_batch = np.expand_dims(img_input, axis=0)
            outputs = self.predict(img_batch)
            res = self.postprocess(outputs, self.beam_search, self.beam_size)

            if res:
                return res[0]  # 返回第一个结果
            else:
                return ("", 0.0)


def main(opt):
    ppocrv2_rec = PPOCRv2Rec(opt)
    img_list = []
    for img_name in os.listdir(opt.input):
        # print(file_name)
        # img_name = '川JK0707.jpg'
        img_file = os.path.join(opt.input, img_name)
        # print(img_file, label)
        src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
        # print(src_img.shape)
        img_list.append(src_img)

    rec_res = ppocrv2_rec(img_list)

    for i, id in enumerate(rec_res.get("ids")):
        logging.info(
            "img_name:{}, conf:{:.6f}, pred:{}".format(
                os.listdir(opt.input)[id], rec_res["res"][i][1], rec_res["res"][i][0]
            )
        )


def img_size_type(arg):
    # 将字符串解析为列表类型
    img_sizes = arg.strip("[]").split("],[")
    img_sizes = [size.split(",") for size in img_sizes]
    img_sizes = [[int(width), int(height)] for width, height in img_sizes]
    return img_sizes


def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument("--dev_id", type=int, default=0, help="tpu card id")
    parser.add_argument(
        "--input",
        type=str,
        default="../datasets/cali_set_rec",
        help="input image directory path",
    )
    parser.add_argument(
        "--bmodel_rec",
        type=str,
        default="../models/BM1684X/ch_PP-OCRv3_rec_fp16.bmodel",
        help="recognizer bmodel path",
    )
    parser.add_argument(
        "--img_size",
        type=img_size_type,
        default=[[640, 48], [320, 48]],
        help="You should set inference size [width,height] manually if using multi-stage bmodel.",
    )
    parser.add_argument(
        "--char_dict_path", type=str, default="../datasets/ppocr_keys_v1.txt"
    )
    parser.add_argument("--use_space_char", type=bool, default=True)
    parser.add_argument(
        "--use_beam_search",
        action="store_const",
        const=True,
        default=False,
        help="Enable beam search",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        choices=range(1, 41),
        help="Only valid when using beam search, valid range 1~40",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

import os

import cv2
import numpy as np

from untool import EngineOV
from tqdm import tqdm

from magic_pdf.libs.config_reader import get_local_models_dir


class DocLayoutYOLOModel(object):
    def __init__(self, weight, device):
        self.model = EngineOV(
            model_path=os.path.join(
                get_local_models_dir(),
                "Layout",
                "doclayout_yolo_docstructbench_imgsz1280_2501_f16.bmodel",
            ),
            device_id=0,
        )
        self.imgsz = 1280

    def preprocess(self, image):
        """
        预处理图像，仿照BasePredictor的preprocess和pre_transform流程

        Args:
            image: 输入图像 (numpy array, HWC格式, BGR)

        Returns:
            tuple: (预处理后的图像, 缩放比例, 填充信息)
        """
        # 应用letterbox变换，类似LetterBox
        img, ratio, (dw, dh) = self._letterbox(
            image, new_shape=(self.imgsz, self.imgsz)
        )

        # 转换为模型输入格式 (类似BasePredictor.preprocess)
        # BGR -> RGB, HWC -> CHW
        img = img[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0  # 0-255 to 0.0-1.0

        return img, ratio, (dw, dh)

    def _letterbox(
        self,
        img,
        new_shape=(1280, 1280),
        color=(114, 114, 114),
        auto=False,
        scaleup=True,
        stride=32,
    ):
        """
        调整图像大小并填充，仿照LetterBox类的实现

        Args:
            img: 输入图像
            new_shape: 目标尺寸
            color: 填充颜色
            auto: 是否自动调整
            scaleup: 是否允许放大
            stride: 步长

        Returns:
            tuple: (调整后的图像, 缩放比例, 填充信息)
        """
        shape = img.shape[:2]  # 当前形状 [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 计算缩放比例 (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # 只缩小，不放大
            r = min(r, 1.0)

        # 计算填充
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # 最小矩形
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2  # 将填充分到两边
        dh /= 2

        if shape[::-1] != new_unpad:  # 需要resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        return img, (r, r), (dw, dh)

    def postprocess(self, results, ratio, pad_info, orig_shape):
        """
        后处理模型输出，将坐标从预处理后的图像映射回原始图像

        Args:
            results: 模型原始输出 (np.array, shape: [1, 300, 6])
                    6个值分别是: [x1, y1, x2, y2, confidence, class_id]
            ratio: 缩放比例 (r, r)
            pad_info: 填充信息 (dw, dh)
            orig_shape: 原始图像形状 (height, width)

        Returns:
            processed_results: 后处理的结果，坐标已映射回原始图像
        """
        layout_res = []
        dw, dh = pad_info
        r = ratio[0]  # 缩放比例

        # 获取第一个batch的结果 [300, 6]
        detections = results[0]
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            
            # 过滤低置信度的检测结果
            if conf < 0.25:  # 可以调整置信度阈值
                continue

            # 坐标转换：从预处理后的图像映射回原始图像
            # 移除padding
            x1 = (x1 - dw) / r
            y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r
            y2 = (y2 - dh) / r

            # 裁剪到原始图像边界
            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))

            xmin, ymin, xmax, ymax = [int(p) for p in [x1, y1, x2, y2]]

            new_item = {
                "category_id": int(cls),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf), 3),
            }
            layout_res.append(new_item)

        return layout_res

    def predict(self, image):
        """
        单图像预测，包含完整的前处理和后处理流程

        Args:
            image: 输入图像 (numpy array)

        Returns:
            layout_res: 检测结果列表
        """
        orig_shape = image.shape[:2]  # 原始图像形状

        # 前处理
        preprocessed_img, ratio, pad_info = self.preprocess(image)

        # 推理
        doclayout_yolo_res = self.model([preprocessed_img])[0]

        # 后处理并格式化输出
        layout_res = self.postprocess(doclayout_yolo_res, ratio, pad_info, orig_shape)

        return layout_res

    def batch_predict(self, images: list, batch_size: int) -> list:
        """
        批量图像预测，包含完整的前处理和后处理流程

        Args:
            images: 输入图像列表
            batch_size: 批处理大小

        Returns:
            images_layout_res: 所有图像的检测结果列表
        """
        images_layout_res = []
        for index in tqdm(range(0, len(images), batch_size), desc="Layout Predict"):
            batch_images = images[index : index + batch_size]

            # 收集原始图像形状
            orig_shapes = [img.shape[:2] for img in batch_images]

            # 批量前处理
            batch_data = [self.preprocess(img) for img in batch_images]
            preprocessed_batch = [data[0] for data in batch_data]
            ratios = [data[1] for data in batch_data]
            pad_infos = [data[2] for data in batch_data]

            # 批量推理
            doclayout_yolo_res = self.model(preprocessed_batch)[0]

            # 批量后处理
            for i, image_res in enumerate(doclayout_yolo_res):
                layout_res = self.postprocess(
                    image_res[np.newaxis, :, :], ratios[i], pad_infos[i], orig_shapes[i]  # 添加batch维度
                )
                images_layout_res.append(layout_res)

        return images_layout_res

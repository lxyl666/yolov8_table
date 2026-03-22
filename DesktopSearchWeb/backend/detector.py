import onnxruntime as ort
import numpy as np
import cv2
from typing import List, Tuple


class YOLODetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.input_size = 416
        self.confidence_threshold = 0.1
        self.iou_threshold = 0.45
        self.class_names = ["key", "usb_disk", "scissors", "paperclip", "earring"]
        self.class_colors = [
            (255, 87, 34),  # key: 深橙红
            (76, 175, 80),  # usb_disk: 深绿
            (33, 150, 243),  # scissors: 深蓝
            (255, 193, 7),  # paperclip: 琥珀金
            (233, 30, 99)  # earring: 洋红
        ]

        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 【关键修复】打印模型输出维度，确认格式
        test_input = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        test_output = self.session.run([self.output_name], {self.input_name: test_input})[0]
        print(f"✅ 模型加载成功，输入名: {self.input_name}, 输出名: {self.output_name}")
        print(f"📊 模型输出维度: {test_output.shape}")

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
        """
        预处理：等比例缩放+填充黑边，和Android端一致
        """
        original_h, original_w = img.shape[:2]

        # 计算缩放比例
        scale = min(self.input_size / original_w, self.input_size / original_h)
        scaled_w = int(original_w * scale)
        scaled_h = int(original_h * scale)

        # 缩放图片
        scaled_img = cv2.resize(img, (scaled_w, scaled_h))

        # 填充黑边到416x416
        input_img = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        offset_x = (self.input_size - scaled_w) // 2
        offset_y = (self.input_size - scaled_h) // 2
        input_img[offset_y:offset_y + scaled_h, offset_x:offset_x + scaled_w] = scaled_img

        # 转成RGB，归一化，转成CHW格式
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = input_img[np.newaxis, :, :, :]

        return input_tensor, scale, offset_x, offset_y, original_w, original_h

    def postprocess(self, output: np.ndarray, scale: float, offset_x: float, offset_y: float, original_w: int,
                    original_h: int) -> List[dict]:
        """
        【核心修复】适配YOLOv8标准输出格式 [1, 9, 3549]
        """
        detections = []

        # 转置输出，变成 [1, 3549, 9]，方便处理
        output = output.transpose(0, 2, 1)
        output = output[0]  # [3549, 9]

        for anchor_idx in range(output.shape[0]):
            # 解析坐标：cx, cy, w, h
            cx, cy, w, h = output[anchor_idx, :4]
            # 解析类别分数：后面的5个值
            class_scores = output[anchor_idx, 4:]

            # 【修复】防止类别索引越界
            if len(class_scores) < len(self.class_names):
                continue

            best_class_idx = np.argmax(class_scores)
            best_conf = class_scores[best_class_idx]

            if best_conf < self.confidence_threshold:
                continue

            # 坐标还原
            x1_in_model = cx - w / 2
            y1_in_model = cy - h / 2
            x2_in_model = cx + w / 2
            y2_in_model = cy + h / 2

            x1_scaled = x1_in_model - offset_x
            y1_scaled = y1_in_model - offset_y
            x2_scaled = x2_in_model - offset_x
            y2_scaled = y2_in_model - offset_y

            x1 = x1_scaled / scale
            y1 = y1_scaled / scale
            x2 = x2_scaled / scale
            y2 = y2_scaled / scale

            # 限制在原图范围内
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))

            if (x2 - x1) > 10 and (y2 - y1) > 10:
                detections.append({
                    "class_id": int(best_class_idx),
                    "class_name": self.class_names[best_class_idx],
                    "confidence": float(best_conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

        # NMS
        return self._nms(detections)

    def _nms(self, detections: List[dict]) -> List[dict]:
        """非极大值抑制"""
        if not detections:
            return []

        # 按置信度排序
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        result = []

        while detections:
            best = detections.pop(0)
            result.append(best)

            # 移除重叠度高的框
            detections = [d for d in detections if self._iou(best["bbox"], d["bbox"]) < self.iou_threshold]

        return result

    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """计算IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection / (area1 + area2 - intersection)

    def detect_image(self, img_path: str) -> Tuple[np.ndarray, List[dict]]:
        """检测图片，返回带框的图片和检测结果"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")

        input_tensor, scale, offset_x, offset_y, original_w, original_h = self.preprocess(img)
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        detections = self.postprocess(output, scale, offset_x, offset_y, original_w, original_h)

        # 画框
        result_img = img.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = self.class_colors[det["class_id"]]
            # 画3D风格框
            self._draw_3d_box(result_img, x1, y1, x2, y2, color)
            # 画标签
            label = f"{det['class_name']} {det['confidence']:.2f}"
            self._draw_label(result_img, label, x1, y1, color)

        return result_img, detections

    def _draw_3d_box(self, img, x1, y1, x2, y2, color):
        """画3D风格框，和Android端一致"""
        box_h = y2 - y1
        perspective_offset = int(box_h * 0.15)

        # 顶面
        top_x1 = x1 + int(perspective_offset * 0.3)
        top_y1 = y1 - perspective_offset
        top_x2 = x2 - int(perspective_offset * 0.3)
        top_y2 = y1

        # 画侧面
        cv2.line(img, (x1, y1), (top_x1, top_y2), color, 2)
        cv2.line(img, (top_x1, top_y2), (top_x1, top_y1), color, 2)
        cv2.line(img, (top_x1, top_y1), (x1, y1), color, 2)
        cv2.line(img, (x2, y1), (top_x2, top_y2), color, 2)
        cv2.line(img, (top_x2, top_y2), (top_x2, top_y1), color, 2)
        cv2.line(img, (top_x2, top_y1), (x2, y1), color, 2)

        # 画顶面
        cv2.rectangle(img, (top_x1, top_y1), (top_x2, top_y2), color, 2)

        # 画正面
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

        # 画角点
        cv2.circle(img, (x1, y1), 8, color, -1)
        cv2.circle(img, (x2, y1), 8, color, -1)
        cv2.circle(img, (x1, y2), 8, color, -1)
        cv2.circle(img, (x2, y2), 8, color, -1)

    def _draw_label(self, img, label, x, y, color):
        """画标签"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 标签背景
        cv2.rectangle(img, (x, y - text_h - 20), (x + text_w + 20, y), (0, 0, 0), -1)
        # 标签文字
        cv2.putText(img, label, (x + 10, y - 5), font, font_scale, color, thickness)
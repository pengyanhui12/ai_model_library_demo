import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class ONNXDetector:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        config_path = self.model_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Cannot find config.json: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model_name = self.config["model_name"]
        self.task = self.config["task"]
        self.input_size = tuple(self.config.get("input_size", [640, 640]))
        self.conf_threshold = float(self.config.get("conf_threshold", 0.25))
        self.iou_threshold = float(self.config.get("iou_threshold", 0.45))

        model_path = self.model_dir / self.config["model_file"]
        if not model_path.exists():
            raise FileNotFoundError(f"Cannot find ONNX model: {model_path}")

        providers = self.config.get("providers", ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(model_path), providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def letterbox(self, image: np.ndarray):
        target_w, target_h = self.input_size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        return padded, scale, left, top

    def preprocess(self, image: np.ndarray):
        img, scale, pad_x, pad_y = self.letterbox(image)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img, scale, pad_x, pad_y

    def nms(self, boxes, scores):
        if len(boxes) == 0:
            return []

        xywh_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            xywh_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

        indices = cv2.dnn.NMSBoxes(
            xywh_boxes,
            scores,
            self.conf_threshold,
            self.iou_threshold
        )

        if len(indices) == 0:
            return []

        return np.array(indices).reshape(-1).tolist()

    def postprocess(self, output, original_shape, scale, pad_x, pad_y):
        pred = output[0]

        # 常见 YOLO ONNX 输出：
        # YOLOv8 / YOLO11: [1, 84, 8400] 或 [1, 8400, 84]
        pred = np.squeeze(pred)

        if pred.ndim != 2:
            raise RuntimeError(f"Unexpected output shape: {pred.shape}")

        # 如果是 [84, 8400]，转成 [8400, 84]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes = []
        scores = []
        class_ids = []

        # 情况1：YOLOv5 风格，输出为 [x, y, w, h, obj, cls...]
        # 情况2：YOLOv8/YOLO11 风格，输出为 [x, y, w, h, cls...]
        has_objectness = pred.shape[1] == 85

        for row in pred:
            if has_objectness:
                obj_score = row[4]
                cls_scores = row[5:]
                class_id = int(np.argmax(cls_scores))
                score = float(obj_score * cls_scores[class_id])
                box = row[:4]
            else:
                cls_scores = row[4:]
                class_id = int(np.argmax(cls_scores))
                score = float(cls_scores[class_id])
                box = row[:4]

            if score < self.conf_threshold:
                continue

            cx, cy, w, h = box

            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale

            img_h, img_w = original_shape[:2]
            x1 = max(0, min(float(x1), img_w - 1))
            y1 = max(0, min(float(y1), img_h - 1))
            x2 = max(0, min(float(x2), img_w - 1))
            y2 = max(0, min(float(y2), img_h - 1))

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(class_id)

        keep = self.nms(boxes, scores)

        results = []
        for idx in keep:
            results.append({
                "class_id": int(class_ids[idx]),
                "confidence": round(float(scores[idx]), 4),
                "bbox": [round(float(v), 2) for v in boxes[idx]]
            })

        return results

    def predict_image(self, image: np.ndarray):
        input_tensor, scale, pad_x, pad_y = self.preprocess(image)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )

        results = self.postprocess(
            outputs,
            image.shape,
            scale,
            pad_x,
            pad_y
        )

        return {
            "model_name": self.model_name,
            "task": self.task,
            "num_detections": len(results),
            "results": results
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model_library/vision/object_detection/yolo_demo")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    detector = ONNXDetector(args.model_dir)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    result = detector.predict_image(image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
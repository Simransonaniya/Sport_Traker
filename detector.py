from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, model='yolov8n.pt', conf_threshold=0.35, iou_threshold=0.45, device='cpu', classes=None):
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.device = device
        self.classes = classes if classes is not None else [0]
        print(f'Loading {model} on {device}...')
        self.model = YOLO(model)
        print('Model loaded!')

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False
        )[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class': cls_name,
                'class_id': cls_id
            })
        return detections

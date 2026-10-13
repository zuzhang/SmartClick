from pathlib import Path

import numpy as np
from ultralytics import YOLO

from detector.base import Detection
from detector.labels import SUPPORTED_LABELS


class YOLODetector:
    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"YOLO model file does not exist: {path}")
        self.model = YOLO(str(path))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model(frame, verbose=False)
        detections = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls.item())
                label = names[class_id]
                if label not in SUPPORTED_LABELS:
                    continue
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf.item()),
                        center_x=round((x1 + x2) / 2),
                        center_y=round((y1 + y2) / 2),
                    )
                )
        return detections

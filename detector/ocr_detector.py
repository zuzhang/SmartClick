from collections.abc import Sequence

import easyocr
import numpy as np

from detector.base import Detection
from detector.labels import label_for_text


class OCRDetector:
    def __init__(self, languages: Sequence[str] = ("en",), gpu: bool = False):
        self.reader = easyocr.Reader(list(languages), gpu=gpu)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections = []
        for bounds, text, confidence in self.reader.readtext(frame):
            label = label_for_text(text)
            if label is None:
                continue
            center_x = round(sum(point[0] for point in bounds) / len(bounds))
            center_y = round(sum(point[1] for point in bounds) / len(bounds))
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidence),
                    center_x=center_x,
                    center_y=center_y,
                )
            )
        return detections

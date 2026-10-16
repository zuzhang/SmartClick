from collections.abc import Iterable

from detector.base import Detection


PRIORITY = {
    "close_button": 100,
    "skip": 90,
    # "claim": 80,
    "collect": 70,
    "confirm": 120,
    # "continue": 50,
    # "next": 40,
    "watch": 130,
    "play": 110,
}


class DecisionEngine:
    def __init__(self, min_confidence: float = 0.8):
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self.min_confidence = min_confidence

    def choose(self, detections: Iterable[Detection]) -> Detection | None:
        eligible = (
            detection
            for detection in detections
            if detection.confidence >= self.min_confidence
        )
        return max(
            eligible,
            key=lambda detection: (
                PRIORITY.get(detection.label, 0),
                detection.confidence,
            ),
            default=None,
        )

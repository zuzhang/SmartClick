from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    center_x: int
    center_y: int


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections in frame-relative pixel coordinates."""

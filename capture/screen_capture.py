from collections.abc import Mapping

import cv2
import mss
import numpy as np


class ScreenCapture:
    def __init__(self):
        self._capture = mss.mss()

    def grab(self, region: Mapping[str, int]) -> np.ndarray:
        screenshot = self._capture.grab(dict(region))
        frame = np.asarray(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._capture.close()

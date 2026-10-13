from collections.abc import Mapping

from detector.base import Detection


class Clicker:
    def __init__(self, region: Mapping[str, int]):
        self.region = dict(region)
        import pyautogui

        self._pyautogui = pyautogui
        self._pyautogui.FAILSAFE = True

    def click_detection(self, detection: Detection) -> tuple[int, int]:
        if not (
            0 <= detection.center_x < self.region["width"]
            and 0 <= detection.center_y < self.region["height"]
        ):
            raise ValueError("Detection point is outside the captured phone region")

        screen_x = self.region["left"] + detection.center_x
        screen_y = self.region["top"] + detection.center_y
        self._pyautogui.click(screen_x, screen_y)
        return screen_x, screen_y

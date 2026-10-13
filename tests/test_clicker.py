import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from actions.clicker import Clicker
from detector.base import Detection


class ClickerTests(unittest.TestCase):
    def setUp(self):
        self.pyautogui = SimpleNamespace(FAILSAFE=False, click=Mock())
        self.patch = patch.dict("sys.modules", pyautogui=self.pyautogui)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_offsets_frame_coordinates_and_clicks_inside_region(self):
        clicker = Clicker({"left": 100, "top": 200, "width": 400, "height": 800})
        point = clicker.click_detection(Detection("skip", 0.9, 25, 50))
        self.assertEqual(point, (125, 250))
        self.pyautogui.click.assert_called_once_with(125, 250)
        self.assertTrue(self.pyautogui.FAILSAFE)

    def test_rejects_points_outside_region(self):
        clicker = Clicker({"left": 100, "top": 200, "width": 400, "height": 800})
        with self.assertRaises(ValueError):
            clicker.click_detection(Detection("skip", 0.9, 400, 50))
        self.pyautogui.click.assert_not_called()

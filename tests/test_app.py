import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import app
from detector.base import Detection


class AppTests(unittest.TestCase):
    def test_interval_is_unchanged_when_no_click_occurs(self):
        detection = Detection("play", 0.9, 10, 20)
        detector = Mock()
        detector.detect.side_effect = [[detection], []]
        decision_engine = Mock()
        decision_engine.choose.side_effect = [detection, None]
        capture = Mock()
        capture.grab.side_effect = [object(), object(), KeyboardInterrupt]
        screen_capture_module = types.ModuleType("capture.screen_capture")
        screen_capture_module.ScreenCapture = Mock(return_value=capture)
        pyautogui_module = types.ModuleType("pyautogui")
        pyautogui_module.FailSafeException = type("FailSafeException", (Exception,), {})

        config = {
            "region": {"left": 0, "top": 0, "width": 100, "height": 100},
            "interval_seconds": 1.0,
            "interval_seconds_by_label": {"play": 3.0},
        }
        with (
            patch.object(app, "load_config", return_value=config),
            patch.object(app, "create_detector", return_value=detector),
            patch.object(app, "DecisionEngine", return_value=decision_engine),
            patch.object(app, "Clicker"),
            patch.object(app.time, "sleep") as sleep,
            patch.object(sys, "argv", ["app.py"]),
            patch.dict(
                sys.modules,
                {
                    "capture.screen_capture": screen_capture_module,
                    "pyautogui": pyautogui_module,
                },
            ),
        ):
            app.main()

        self.assertEqual(sleep.call_args_list, [call(3.0), call(3.0)])
        capture.close.assert_called_once()

    def test_saves_once_after_timeout_and_resets_after_click(self):
        detection = Detection("play", 0.9, 10, 20)
        frames = [object() for _ in range(5)]
        detector = Mock()
        detector.detect.side_effect = [
            [detection],
            [],
            [],
            [detection],
            [],
        ]
        decision_engine = Mock()
        decision_engine.choose.side_effect = [
            detection,
            None,
            None,
            detection,
            None,
        ]
        capture = Mock()
        capture.grab.side_effect = [*frames, KeyboardInterrupt]
        screen_capture_module = types.ModuleType("capture.screen_capture")
        screen_capture_module.ScreenCapture = Mock(return_value=capture)
        opencv_module = types.ModuleType("cv2")
        opencv_module.imwrite = Mock(return_value=True)
        pyautogui_module = types.ModuleType("pyautogui")
        pyautogui_module.FailSafeException = type("FailSafeException", (Exception,), {})

        with tempfile.TemporaryDirectory() as directory:
            config = {
                "region": {"left": 0, "top": 0, "width": 100, "height": 100},
                "no_click_screenshot_after_seconds": 1.0,
                "no_click_screenshot_dir": directory,
            }
            with (
                patch.object(app, "load_config", return_value=config),
                patch.object(app, "create_detector", return_value=detector),
                patch.object(app, "DecisionEngine", return_value=decision_engine),
                patch.object(app, "Clicker"),
                patch.object(
                    app.time,
                    "monotonic",
                    side_effect=[0.0, 0.0, 0.0, 1.1, 5.0, 6.1],
                ),
                patch.object(app.time, "sleep"),
                patch.object(sys, "argv", ["app.py"]),
                patch.dict(
                    sys.modules,
                    {
                        "capture.screen_capture": screen_capture_module,
                        "cv2": opencv_module,
                        "pyautogui": pyautogui_module,
                    },
                ),
            ):
                app.main()

        self.assertEqual(opencv_module.imwrite.call_count, 2)
        self.assertEqual(
            [
                Path(call.args[0]).parent
                for call in opencv_module.imwrite.call_args_list
            ],
            [Path(directory), Path(directory)],
        )
        self.assertEqual(
            [call.args[1] for call in opencv_module.imwrite.call_args_list],
            [frames[1], frames[4]],
        )
        capture.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import app
from detector.base import Detection


class KeyboardListener:
    def __init__(self, on_press):
        self.on_press = on_press

    def start(self):
        pass

    def stop(self):
        pass


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
        pynput_module = types.ModuleType("pynput")
        pynput_module.keyboard = types.SimpleNamespace(
            Listener=KeyboardListener,
            Key=types.SimpleNamespace(f8=object()),
        )

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
            patch.object(app, "_save_f8_screenshots") as save_f8_screenshots,
            patch.object(sys, "argv", ["app.py"]),
            patch.dict(
                sys.modules,
                {
                    "capture.screen_capture": screen_capture_module,
                    "pyautogui": pyautogui_module,
                    "pynput": pynput_module,
                },
            ),
        ):
            app.main()

        self.assertEqual(
            [
                call.kwargs.get("wait_timeout", 0)
                for call in save_f8_screenshots.call_args_list
                if call.kwargs
            ],
            [3.0, 3.0],
        )
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
        pynput_module = types.ModuleType("pynput")
        pynput_module.keyboard = types.SimpleNamespace(
            Listener=KeyboardListener,
            Key=types.SimpleNamespace(f8=object()),
        )

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
                patch.object(app, "_save_f8_screenshots"),
                patch.object(sys, "argv", ["app.py"]),
                patch.dict(
                    sys.modules,
                    {
                        "capture.screen_capture": screen_capture_module,
                        "cv2": opencv_module,
                        "pyautogui": pyautogui_module,
                        "pynput": pynput_module,
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

    def test_f8_saves_current_region_screenshot(self):
        monitored_frame = object()
        manual_frame = object()
        capture = Mock()
        capture.grab.side_effect = [
            monitored_frame,
            manual_frame,
            KeyboardInterrupt,
        ]

        class TriggeringKeyboardListener(KeyboardListener):
            def start(self):
                self.on_press(keyboard_module.keyboard.Key.f8)

        screen_capture_module = types.ModuleType("capture.screen_capture")
        screen_capture_module.ScreenCapture = Mock(return_value=capture)
        opencv_module = types.ModuleType("cv2")
        opencv_module.imwrite = Mock(return_value=True)
        pyautogui_module = types.ModuleType("pyautogui")
        pyautogui_module.FailSafeException = type("FailSafeException", (Exception,), {})
        keyboard_module = types.ModuleType("pynput")
        keyboard_module.keyboard = types.SimpleNamespace(
            Listener=TriggeringKeyboardListener,
            Key=types.SimpleNamespace(f8=object()),
        )
        detector = Mock()
        detector.detect.return_value = []
        decision_engine = Mock()
        decision_engine.choose.return_value = None

        with tempfile.TemporaryDirectory() as directory:
            config = {
                "region": {"left": 10, "top": 20, "width": 100, "height": 100},
                "interval_seconds": 0.001,
                "no_click_screenshot_dir": directory,
            }
            with (
                patch.object(app, "load_config", return_value=config),
                patch.object(app, "create_detector", return_value=detector),
                patch.object(app, "DecisionEngine", return_value=decision_engine),
                patch.object(app, "Clicker"),
                patch.object(sys, "argv", ["app.py"]),
                patch.dict(
                    sys.modules,
                    {
                        "capture.screen_capture": screen_capture_module,
                        "cv2": opencv_module,
                        "pyautogui": pyautogui_module,
                        "pynput": keyboard_module,
                    },
                ),
            ):
                app.main()

        opencv_module.imwrite.assert_called_once()
        saved_path, saved_frame = opencv_module.imwrite.call_args.args
        self.assertTrue(Path(saved_path).name.startswith("f8_"))
        self.assertEqual(saved_frame, manual_frame)
        capture.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

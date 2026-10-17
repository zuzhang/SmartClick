import sys
import types
import unittest
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
            patch.dict(sys.modules, {"capture.screen_capture": screen_capture_module}),
        ):
            app.main()

        self.assertEqual(sleep.call_args_list, [call(3.0), call(3.0)])
        capture.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

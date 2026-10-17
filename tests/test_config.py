import tempfile
import unittest
from pathlib import Path

from app import _interval_for_label, load_config


class ConfigTests(unittest.TestCase):
    def test_loads_valid_config(self):
        contents = """
detector: ocr
region:
  left: 10
  top: 20
  width: 300
  height: 600
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(contents, encoding="utf-8")
            self.assertEqual(load_config(str(path))["region"]["width"], 300)

    def test_rejects_non_positive_region_dimensions(self):
        contents = """
detector: ocr
region:
  left: 10
  top: 20
  width: 0
  height: 600
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(contents, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(str(path))

    def test_loads_yolo_device(self):
        contents = """
detector: yolo
model_path: models/best.pt
device: mps
region:
  left: 10
  top: 20
  width: 300
  height: 600
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(contents, encoding="utf-8")
            self.assertEqual(load_config(str(path))["device"], "mps")

    def test_rejects_invalid_device(self):
        contents = """
device: " "
region:
  left: 10
  top: 20
  width: 300
  height: 600
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(contents, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(str(path))

    def test_loads_per_label_intervals(self):
        contents = """
region:
  left: 10
  top: 20
  width: 300
  height: 600
interval_seconds_by_label:
  close_button: 1.0
  skip: 1.0
  play: 3.0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(contents, encoding="utf-8")
            self.assertEqual(
                load_config(str(path))["interval_seconds_by_label"]["play"],
                3.0,
            )

    def test_rejects_invalid_per_label_interval(self):
        contents = """
region:
  left: 10
  top: 20
  width: 300
  height: 600
interval_seconds_by_label:
  play: 0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(contents, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(str(path))

    def test_rejects_invalid_no_click_screenshot_timeout(self):
        contents = """
region:
  left: 10
  top: 20
  width: 300
  height: 600
no_click_screenshot_after_seconds: 0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(contents, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(str(path))

    def test_rejects_empty_no_click_screenshot_directory(self):
        contents = """
region:
  left: 10
  top: 20
  width: 300
  height: 600
no_click_screenshot_dir: " "
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(contents, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(str(path))

    def test_uses_label_interval_or_global_fallback(self):
        intervals = {"close_button": 1.0, "skip": 1.0, "play": 3.0}
        self.assertEqual(_interval_for_label("close_button", 2.0, intervals), 1.0)
        self.assertEqual(_interval_for_label("skip", 2.0, intervals), 1.0)
        self.assertEqual(_interval_for_label("play", 2.0, intervals), 3.0)
        self.assertEqual(_interval_for_label("claim", 2.0, intervals), 2.0)
        self.assertEqual(_interval_for_label(None, 2.0, intervals), 2.0)


if __name__ == "__main__":
    unittest.main()

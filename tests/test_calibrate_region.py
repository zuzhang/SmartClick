import tempfile
import unittest
from pathlib import Path

from calibrate_region import _selected_region, save_region


class RegionCalibrationTests(unittest.TestCase):
    def test_maps_selection_to_screen_coordinates(self):
        self.assertEqual(
            _selected_region(
                {"left": 0, "top": 0, "width": 1920, "height": 1080},
                (120, 80, 430, 900),
            ),
            {"left": 120, "top": 80, "width": 430, "height": 900},
        )

    def test_includes_monitor_offset(self):
        self.assertEqual(
            _selected_region(
                {"left": 100, "top": 50, "width": 1920, "height": 1080},
                (120, 80, 430, 900),
            ),
            {"left": 220, "top": 130, "width": 430, "height": 900},
        )

    def test_rejects_empty_selection(self):
        with self.assertRaisesRegex(ValueError, "No screen region"):
            _selected_region(
                {"left": 0, "top": 0, "width": 1920, "height": 1080},
                (0, 0, 0, 0),
            )

    def test_updates_only_region_values_and_preserves_comments(self):
        contents = """# Keep this comment.
region:
  left: 10 # screen x
  top: 20
  width: 300
  height: 600
detector: ocr
"""
        region = {"left": 100, "top": 200, "width": 430, "height": 900}
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(contents, encoding="utf-8")

            save_region(config_path, region)

            updated = config_path.read_text(encoding="utf-8")
            self.assertIn("# Keep this comment.", updated)
            self.assertIn("left: 100 # screen x", updated)
            self.assertIn("detector: ocr", updated)
            self.assertIn("height: 900", updated)


if __name__ == "__main__":
    unittest.main()

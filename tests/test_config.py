import tempfile
import unittest
from pathlib import Path

from app import load_config


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


if __name__ == "__main__":
    unittest.main()

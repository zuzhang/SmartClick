import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from training import prepare_cvat_predictions as prepare


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    names = {0: "close_button", 1: "play"}

    def __init__(self):
        self.confidences = []

    def predict(self, source, conf, verbose):
        self.confidences.append(conf)
        if Path(source).name == "screen01.png":
            boxes = SimpleNamespace(
                cls=FakeTensor([1]),
                xywhn=FakeTensor([[0.5, 0.4, 0.2, 0.1]]),
            )
        else:
            boxes = SimpleNamespace(
                cls=FakeTensor([]),
                xywhn=FakeTensor([]),
            )
        return [SimpleNamespace(boxes=boxes, path=source)]


class PrepareCvatPredictionsTests(unittest.TestCase):
    def test_example_config_uses_requested_defaults(self):
        config = prepare._read_config(prepare.EXAMPLE_CONFIG)

        self.assertEqual(config["model_path"], "models/best.pt")
        self.assertEqual(config["confidence"], 0.2)
        self.assertEqual(config["source_dir"], "review_screenshots")

    def test_archive_contains_originals_predictions_and_model_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screenshots = root / "screenshots"
            screenshots.mkdir()
            (screenshots / "screen01.png").write_bytes(b"original screenshot")
            (screenshots / "screen02.png").write_bytes(b"another screenshot")
            model_path = root / "best.pt"
            model_path.write_bytes(b"test model")
            output_dir = root / "output"
            model = FakeModel()

            with patch.object(prepare, "load_model", return_value=model):
                archive_path = prepare.create_cvat_archive(
                    screenshots,
                    model_path,
                    0.2,
                    output_dir,
                )

            self.assertEqual(model.confidences, [0.2, 0.2])
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {
                        "data.yaml",
                        "train.txt",
                        "images/train/screen01.png",
                        "images/train/screen02.png",
                        "labels/train/screen01.txt",
                        "labels/train/screen02.txt",
                    },
                )
                self.assertEqual(
                    archive.read("images/train/screen01.png"),
                    b"original screenshot",
                )
                self.assertEqual(
                    archive.read("train.txt").decode(),
                    "images/train/screen01.png\nimages/train/screen02.png\n",
                )
                self.assertEqual(
                    archive.read("labels/train/screen01.txt").decode(),
                    "1 0.500000 0.400000 0.200000 0.100000\n",
                )
                self.assertEqual(archive.read("labels/train/screen02.txt"), b"")
                dataset_config = yaml.safe_load(archive.read("data.yaml"))
                self.assertEqual(dataset_config["train"], "train.txt")
                self.assertEqual(
                    dataset_config["names"],
                    {0: "close_button", 1: "play"},
                )

    def test_rejects_duplicate_filenames_in_nested_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screenshots = root / "screenshots"
            (screenshots / "first").mkdir(parents=True)
            (screenshots / "second").mkdir()
            (screenshots / "first" / "screen.png").touch()
            (screenshots / "second" / "screen.png").touch()
            model_path = root / "best.pt"
            model_path.touch()

            with self.assertRaisesRegex(ValueError, "filenames must be unique"):
                prepare.create_cvat_archive(
                    screenshots,
                    model_path,
                    0.2,
                    root / "output",
                )

    def test_rejects_confidence_outside_range(self):
        config = {**prepare.DEFAULTS, "confidence": 1.1}

        with self.assertRaisesRegex(ValueError, "confidence must be"):
            prepare._validate_config(config)


if __name__ == "__main__":
    unittest.main()

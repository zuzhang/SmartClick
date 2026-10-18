import tempfile
import unittest
import zipfile
from pathlib import Path

from training.import_cvat import import_dataset


class CvatImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "game-controls.yaml"
        self.config_path.write_text(
f"""
path: {self.root / "training"}
train: images/Train
val: images/Validation
names:
  0: close_button
""",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_export(self, root: Path) -> None:
        for split in ("train", "Validation"):
            images = root / "images" / split
            labels = root / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / f"{split}.png").write_bytes(b"image")
            (labels / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def test_imports_lowercase_train_into_configured_case(self):
        export = self.root / "export"
        self.make_export(export)

        self.assertEqual(import_dataset(export, self.config_path), (2, 2))
        self.assertTrue((self.root / "training/images/Train/train.png").is_file())
        self.assertTrue(
            (self.root / "training/labels/Validation/Validation.txt").is_file()
        )
        self.assertEqual(
            self.config_path.read_text(encoding="utf-8").splitlines()[2],
            "train: images/Train",
        )

    def test_imports_zip_with_wrapped_dataset_folder(self):
        export = self.root / "export"
        self.make_export(export / "cvat-export")
        archive = self.root / "export.zip"
        with zipfile.ZipFile(archive, "w") as zip_file:
            for file in (export / "cvat-export").rglob("*"):
                if file.is_file():
                    zip_file.write(file, file.relative_to(export))

        self.assertEqual(import_dataset(archive, self.config_path), (2, 2))
        self.assertTrue((self.root / "training/images/Train/train.png").is_file())

    def test_collision_fails_without_overwrite(self):
        export = self.root / "export"
        self.make_export(export)
        destination = self.root / "training/images/Train/train.png"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"existing")

        with self.assertRaises(FileExistsError):
            import_dataset(export, self.config_path)
        self.assertEqual(destination.read_bytes(), b"existing")

    def test_overwrite_replaces_matching_files_when_requested(self):
        export = self.root / "export"
        self.make_export(export)
        destination = self.root / "training/images/Train/train.png"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"existing")

        import_dataset(export, self.config_path, overwrite=True)
        self.assertEqual(destination.read_bytes(), b"image")

    def test_rejects_archive_path_traversal(self):
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as zip_file:
            zip_file.writestr("../outside.txt", "unsafe")

        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            import_dataset(archive, self.config_path)


if __name__ == "__main__":
    unittest.main()

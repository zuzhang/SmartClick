import argparse
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import yaml


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
}


def _extract_zip_safely(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            member = PurePosixPath(entry.filename)
            if (
                member.is_absolute()
                or ".." in member.parts
                or "\\" in entry.filename
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe path in CVAT archive: {entry.filename}")
        archive.extractall(destination)


def _source_directory(source: Path, temporary_directory: Path) -> Path:
    if source.is_dir():
        return source
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError("CVAT source must be a ZIP archive or an extracted folder")
    _extract_zip_safely(source, temporary_directory)
    return temporary_directory


def _find_dataset_root(source: Path) -> Path:
    candidates = [
        images.parent
        for images in source.rglob("images")
        if images.is_dir() and (images.parent / "labels").is_dir()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected one CVAT dataset root containing sibling images/ and labels/ folders"
        )
    return candidates[0]


def _configured_directories(
    config: dict,
) -> tuple[dict[str, tuple[Path, Path]], dict[str, str]]:
    dataset_path = Path(config.get("path", "."))
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path

    configured: dict[str, tuple[Path, Path]] = {}
    for key, split in (("train", "train"), ("val", "validation")):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Dataset config must define {key} as an image directory")
        relative_path = Path(value)
        if relative_path.is_absolute():
            image_directory = relative_path
            parts = relative_path.parts
        else:
            image_directory = dataset_path / relative_path
            parts = relative_path.parts
        if relative_path.suffix or "images" not in parts:
            raise ValueError(
                f"Dataset config {key} must point to an images/ split directory"
            )
        if parts.count("images") != 1:
            raise ValueError(f"Dataset config {key} must contain one images/ directory")
        label_parts = list(parts)
        label_parts[label_parts.index("images")] = "labels"
        label_directory = (
            Path(*label_parts)
            if relative_path.is_absolute()
            else dataset_path / Path(*label_parts)
        )
        configured[split] = (image_directory, label_directory)

    if configured["train"][0] == configured["validation"][0]:
        raise ValueError("Dataset config train and val directories must be different")

    split_names = {
        "train": configured["train"][0].name,
        "validation": configured["validation"][0].name,
    }
    return configured, split_names


def _source_files(
    dataset_root: Path,
    split_names: dict[str, str],
) -> dict[str, tuple[list[Path], list[Path]]]:
    files: dict[str, tuple[list[Path], list[Path]]] = {}
    for kind in ("images", "labels"):
        for split_directory in (dataset_root / kind).iterdir():
            if not split_directory.is_dir():
                continue
            normalized_split = SPLIT_ALIASES.get(split_directory.name.lower())
            if normalized_split is None:
                raise ValueError(
                    f"Unsupported CVAT subset folder: {split_directory.name}"
                )
            if normalized_split not in files:
                files[normalized_split] = ([], [])
            index = 0 if kind == "images" else 1
            selected_files = (
                path
                for path in split_directory.rglob("*")
                if path.is_file()
                and (
                    path.suffix.lower() in IMAGE_EXTENSIONS
                    if kind == "images"
                    else path.suffix.lower() == ".txt"
                )
            )
            files[normalized_split][index].extend(selected_files)

    if "train" not in files or "validation" not in files:
        raise ValueError("CVAT export must contain both train and validation subsets")

    for split, (images, labels) in files.items():
        image_stems = [path.stem for path in images]
        label_stems = [path.stem for path in labels]
        if len(image_stems) != len(set(image_stems)):
            raise ValueError(f"Duplicate image filenames in CVAT {split_names[split]} subset")
        if len(label_stems) != len(set(label_stems)):
            raise ValueError(f"Duplicate label filenames in CVAT {split_names[split]} subset")
        missing_images = set(label_stems) - set(image_stems)
        if missing_images:
            names = ", ".join(sorted(missing_images)[:5])
            raise ValueError(
                f"CVAT {split_names[split]} labels have no matching images: {names}"
            )
    return files


def import_dataset(
    source: Path,
    config_path: Path,
    overwrite: bool = False,
) -> tuple[int, int]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Dataset config must contain a YAML mapping")

    configured, split_names = _configured_directories(config)
    with tempfile.TemporaryDirectory(prefix="smartclick-cvat-") as temporary:
        source_root = _source_directory(source, Path(temporary))
        dataset_root = _find_dataset_root(source_root)
        files = _source_files(dataset_root, split_names)

        copy_jobs: list[tuple[Path, Path]] = []
        for split, (images, labels) in files.items():
            image_directory, label_directory = configured[split]
            copy_jobs.extend((path, image_directory / path.name) for path in images)
            copy_jobs.extend((path, label_directory / path.name) for path in labels)

        collisions = [
            destination for _, destination in copy_jobs if destination.exists()
        ]
        if collisions and not overwrite:
            examples = ", ".join(str(path) for path in collisions[:5])
            raise FileExistsError(
                f"Import would overwrite existing files: {examples}. "
                "Use --overwrite to replace them."
            )

        for source_file, destination in copy_jobs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)

        image_count = sum(len(images) for images, _ in files.values())
        label_count = sum(len(labels) for _, labels in files.values())
        return image_count, label_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import a CVAT Ultralytics YOLO export into the configured training dataset."
    )
    parser.add_argument("source", type=Path, help="CVAT ZIP archive or extracted dataset folder")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("game-controls.yaml"),
        help="Dataset YAML (default: training/game-controls.yaml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files with matching names in the target dataset",
    )
    args = parser.parse_args(argv)

    try:
        image_count, label_count = import_dataset(
            args.source,
            args.config,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, zipfile.BadZipFile, yaml.YAMLError) as error:
        print(f"Import failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Imported {image_count} images and {label_count} labels. "
        "The existing dataset YAML was not changed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import math
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "auto_label.yaml"
EXAMPLE_CONFIG = PROJECT_ROOT / "config" / "auto_label.example.yaml"
DEFAULTS = {
    "source_dir": "review_screenshots",
    "model_path": "models/best.pt",
    "confidence": 0.2,
    "output_dir": "cvat_imports",
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    unknown_keys = config.keys() - DEFAULTS.keys()
    if unknown_keys:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown_keys))}")
    return {**DEFAULTS, **config}


def _resolve_path(value: str, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    confidence = config["confidence"]
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("confidence must be a number between 0 and 1")
    return {
        "source_dir": _resolve_path(config["source_dir"], "source_dir"),
        "model_path": _resolve_path(config["model_path"], "model_path"),
        "confidence": confidence,
        "output_dir": _resolve_path(config["output_dir"], "output_dir"),
    }


def load_model(model_path: Path):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        if error.name == "ultralytics":
            raise RuntimeError(
                "Ultralytics is required. Install it with: "
                "python -m pip install ultralytics"
            ) from error
        raise
    return YOLO(str(model_path))


def _model_names(model: Any) -> dict[int, str]:
    raw_names = model.names
    if isinstance(raw_names, list):
        names = dict(enumerate(raw_names))
    elif isinstance(raw_names, dict):
        try:
            names = {int(class_id): name for class_id, name in raw_names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Model class IDs must be integers starting at 0") from error
    else:
        raise ValueError("Model names must be a list or a mapping of class IDs to names")

    if (
        not names
        or set(names) != set(range(len(names)))
        or any(not isinstance(name, str) or not name.strip() for name in names.values())
    ):
        raise ValueError("Model class names must be non-empty and use contiguous IDs from 0")
    return names


def _label_text(result: Any, names: dict[int, str]) -> str:
    boxes = result.boxes
    if boxes is None:
        return ""

    class_ids = boxes.cls.cpu().tolist()
    coordinates = boxes.xywhn.cpu().tolist()
    if len(class_ids) != len(coordinates):
        raise ValueError(f"Prediction returned mismatched boxes for {result.path}")

    lines = []
    for class_id, values in zip(class_ids, coordinates):
        class_id = int(class_id)
        if class_id not in names or len(values) != 4:
            raise ValueError(f"Prediction returned an invalid box for {result.path}")
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"Prediction returned non-finite coordinates for {result.path}")
        lines.append(
            f"{class_id} " + " ".join(f"{value:.6f}" for value in values)
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _image_files(source_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(source_dir).as_posix().lower(),
    )


def create_cvat_archive(
    source_dir: Path,
    model_path: Path,
    confidence: float,
    output_dir: Path,
) -> Path:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Screenshot directory does not exist: {source_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model file does not exist: {model_path}")

    images = _image_files(source_dir)
    if not images:
        raise FileNotFoundError(f"No supported image files found in: {source_dir}")

    image_names = [image.name.lower() for image in images]
    if len(image_names) != len(set(image_names)):
        raise ValueError(
            "Screenshot filenames must be unique (ignoring case) for CVAT import"
        )
    label_names = [image.stem.lower() for image in images]
    if len(label_names) != len(set(label_names)):
        raise ValueError(
            "Screenshot filenames must have unique stems for YOLO label matching"
        )

    model = load_model(model_path)
    names = _model_names(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_path = output_dir / f"cvat_predictions_{timestamp}.zip"

    with tempfile.TemporaryDirectory(prefix=".cvat_predictions_", dir=output_dir) as temp:
        dataset_dir = Path(temp)
        image_dir = dataset_dir / "images" / "train"
        label_dir = dataset_dir / "labels" / "train"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        for image in images:
            shutil.copy2(image, image_dir / image.name)
            results = model.predict(
                source=str(image),
                conf=confidence,
                verbose=False,
            )
            if len(results) != 1:
                raise ValueError(f"Expected one prediction result for {image}")
            label_path = label_dir / f"{image.stem}.txt"
            label_path.write_text(_label_text(results[0], names), encoding="utf-8")

        dataset_config = {
            "path": "./",
            "train": "train.txt",
            "names": names,
        }
        (dataset_dir / "data.yaml").write_text(
            yaml.safe_dump(dataset_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (dataset_dir / "train.txt").write_text(
            "".join(f"images/train/{image.name}\n" for image in images),
            encoding="utf-8",
        )

        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path in sorted(dataset_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(dataset_dir).as_posix())
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict labels for screenshots and package them for CVAT."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config file (default: config/auto_label.yaml, then the example config)",
    )
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else EXAMPLE_CONFIG
    elif not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    config = _validate_config(_read_config(config_path))
    archive_path = create_cvat_archive(**config)
    print(f"Created CVAT dataset archive: {archive_path}")


if __name__ == "__main__":
    main()

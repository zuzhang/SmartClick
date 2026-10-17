import argparse
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from actions.clicker import Clicker
from actions.decision_engine import DecisionEngine


def _interval_for_label(
    label: str | None,
    default_interval: int | float,
    label_intervals: Mapping[str, int | float],
) -> int | float:
    if label is None:
        return default_interval
    return label_intervals.get(label, default_interval)


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")

    region = config.get("region")
    required_region_keys = {"left", "top", "width", "height"}
    if not isinstance(region, dict) or not required_region_keys.issubset(region):
        raise ValueError("region must define left, top, width, and height")
    if any(
        not isinstance(region[key], int) or isinstance(region[key], bool)
        for key in required_region_keys
    ):
        raise ValueError("region coordinates and dimensions must be integers")
    if region["left"] < 0 or region["top"] < 0:
        raise ValueError("region left and top must be non-negative")
    if region["width"] <= 0 or region["height"] <= 0:
        raise ValueError("region width and height must be positive")

    confidence = config.get("min_confidence", 0.8)
    if (
        not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("min_confidence must be between 0 and 1")
    interval = config.get("interval_seconds", 1.0)
    if (
        not isinstance(interval, (int, float))
        or isinstance(interval, bool)
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("interval_seconds must be positive")
    label_intervals = config.get("interval_seconds_by_label", {})
    if not isinstance(label_intervals, dict) or any(
        not isinstance(label, str)
        or not label
        or not isinstance(label_interval, (int, float))
        or isinstance(label_interval, bool)
        or not math.isfinite(label_interval)
        or label_interval <= 0
        for label, label_interval in label_intervals.items()
    ):
        raise ValueError(
            "interval_seconds_by_label must map labels to positive intervals"
        )
    detector_name = config.get("detector", "ocr")
    if detector_name not in {"ocr", "yolo"}:
        raise ValueError("detector must be either 'ocr' or 'yolo'")
    device = config.get("device")
    if device is not None and (
        isinstance(device, bool)
        or not isinstance(device, (str, int))
        or (isinstance(device, str) and not device.strip())
        or (isinstance(device, int) and device < 0)
    ):
        raise ValueError(
            "device must be a non-empty device name or non-negative index"
        )
    if detector_name == "yolo" and not config.get("model_path"):
        raise ValueError("model_path is required when detector is 'yolo'")
    if detector_name == "ocr":
        languages = config.get("languages", ["en"])
        if not isinstance(languages, list) or not languages or any(
            not isinstance(language, str) or not language for language in languages
        ):
            raise ValueError("languages must be a non-empty list of language codes")
        if not isinstance(config.get("gpu", False), bool):
            raise ValueError("gpu must be a boolean")
    return config


def create_detector(config: dict[str, Any]):
    if config.get("detector", "ocr") == "ocr":
        from detector.ocr_detector import OCRDetector

        return OCRDetector(
            languages=config.get("languages", ["en"]),
            gpu=config.get("gpu", False),
        )
    from detector.yolo_detector import YOLODetector

    return YOLODetector(
        model_path=config["model_path"],
        device=config.get("device"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate actions on iPhone Mirroring")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    region = config["region"]
    detector = create_detector(config)
    decision_engine = DecisionEngine(config.get("min_confidence", 0.8))
    clicker = Clicker(region)
    import pyautogui

    from capture.screen_capture import ScreenCapture

    capture = ScreenCapture()
    interval = config.get("interval_seconds", 1.0)
    label_intervals = config.get("interval_seconds_by_label", {})

    try:
        print(
            f"SmartClick running on region {region}; press Ctrl+C or move the "
            "pointer to the screen's upper-left corner to stop."
        )
        while True:
            frame = capture.grab(region)
            detections = detector.detect(frame)
            for detection in detections:
                screen_x = region["left"] + detection.center_x
                screen_y = region["top"] + detection.center_y
                print(
                    f"Detected {detection.label} at screen ({screen_x}, {screen_y}) "
                    f"[region ({detection.center_x}, {detection.center_y}); "
                    f"confidence {detection.confidence:.2f}]"
                )

            target = decision_engine.choose(detections)
            if target is not None:
                point = clicker.click_detection(target)
                print(
                    f"Clicked {target.label} at screen {point} "
                    f"(confidence {target.confidence:.2f})"
                )
            elif detections:
                print(
                    "No detected controls met "
                    f"min_confidence={decision_engine.min_confidence:.2f}; not clicking."
                )
            else:
                print("No recognized controls detected; not clicking.")
            time.sleep(
                _interval_for_label(
                    target.label if target is not None else None,
                    interval,
                    label_intervals,
                )
            )
    except pyautogui.FailSafeException:
        print("\nPyAutoGUI fail-safe activated; SmartClick stopped.")
    except KeyboardInterrupt:
        print("\nSmartClick stopped.")
    finally:
        capture.close()


if __name__ == "__main__":
    main()

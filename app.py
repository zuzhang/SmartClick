import argparse
import math
import queue
import time
from collections.abc import Mapping
from datetime import datetime
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


def _process_keyboard_events(
    keyboard_events: queue.Queue[str],
    capture,
    region: Mapping[str, int],
    screenshot_dir: Path,
    paused: bool,
    wait_timeout: int | float | None = 0,
) -> tuple[bool, bool]:
    resumed = False
    try:
        event = keyboard_events.get(timeout=wait_timeout)
    except queue.Empty:
        return paused, resumed

    while True:
        if event == "screenshot":
            frame = capture.grab(region)
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_dir / (
                f"f8_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            )
            import cv2

            if not cv2.imwrite(str(screenshot_path), frame):
                raise OSError(f"Failed to save F8 screenshot: {screenshot_path}")
            print(f"Saved F8 screenshot to {screenshot_path}")
        elif event == "toggle-pause":
            paused = not paused
            if paused:
                print("Detection paused. Press F9 to resume.")
            else:
                resumed = True
                print("Detection resumed.")
        try:
            event = keyboard_events.get_nowait()
        except queue.Empty:
            return paused, resumed


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
    no_click_screenshot_after = config.get(
        "no_click_screenshot_after_seconds",
        60.0,
    )
    if (
        not isinstance(no_click_screenshot_after, (int, float))
        or isinstance(no_click_screenshot_after, bool)
        or not math.isfinite(no_click_screenshot_after)
        or no_click_screenshot_after <= 0
    ):
        raise ValueError("no_click_screenshot_after_seconds must be positive")
    no_click_screenshot_dir = config.get(
        "no_click_screenshot_dir",
        "review_screenshots",
    )
    if (
        not isinstance(no_click_screenshot_dir, str)
        or not no_click_screenshot_dir.strip()
    ):
        raise ValueError("no_click_screenshot_dir must be a non-empty path")
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
    from pynput import keyboard

    from capture.screen_capture import ScreenCapture

    capture = ScreenCapture()
    default_interval = config.get("interval_seconds", 1.0)
    interval = default_interval
    label_intervals = config.get("interval_seconds_by_label", {})
    no_click_screenshot_after = config.get(
        "no_click_screenshot_after_seconds",
        60.0,
    )
    no_click_screenshot_dir = Path(
        config.get("no_click_screenshot_dir", "review_screenshots")
    )
    no_click_since = time.monotonic()
    no_click_screenshot_saved = False
    keyboard_events: queue.Queue[str] = queue.Queue()

    def on_key_press(key) -> None:
        if key == keyboard.Key.f8:
            keyboard_events.put("screenshot")
        elif key == keyboard.Key.f9:
            keyboard_events.put("toggle-pause")

    try:
        keyboard_listener = keyboard.Listener(on_press=on_key_press)
        keyboard_listener.start()
        print(
            f"SmartClick running on region {region}; press Ctrl+C or move the "
            "pointer to the screen's upper-left corner to stop; press F8 to "
            "save a screenshot and F9 to pause or resume detection."
        )
        paused = False
        while True:
            paused, resumed = _process_keyboard_events(
                keyboard_events,
                capture,
                region,
                no_click_screenshot_dir,
                paused,
            )
            if resumed:
                no_click_since = time.monotonic()
            while paused:
                paused, resumed = _process_keyboard_events(
                    keyboard_events,
                    capture,
                    region,
                    no_click_screenshot_dir,
                    paused,
                    wait_timeout=None,
                )
                if resumed:
                    no_click_since = time.monotonic()
            if resumed:
                continue

            frame = capture.grab(region)
            detections = detector.detect(frame)
            paused, resumed = _process_keyboard_events(
                keyboard_events,
                capture,
                region,
                no_click_screenshot_dir,
                paused,
            )
            if resumed:
                no_click_since = time.monotonic()
            if paused:
                continue

            for detection in detections:
                screen_x = region["left"] + detection.center_x
                screen_y = region["top"] + detection.center_y
                print(
                    f"Detected {detection.label} at screen ({screen_x}, {screen_y}) "
                    f"[region ({detection.center_x}, {detection.center_y}); "
                    f"confidence {detection.confidence:.2f}]"
                )

            target = decision_engine.choose(detections)
            if (
                not no_click_screenshot_saved
                and time.monotonic() - no_click_since
                >= no_click_screenshot_after
            ):
                no_click_screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = no_click_screenshot_dir / (
                    f"{datetime.now():%Y%m%d_%H%M%S_%f}.png"
                )
                import cv2

                if not cv2.imwrite(str(screenshot_path), frame):
                    raise OSError(f"Failed to save review screenshot: {screenshot_path}")
                no_click_screenshot_saved = True
                print(f"Saved no-click review screenshot to {screenshot_path}")

            if target is not None:
                point = clicker.click_detection(target)
                interval = _interval_for_label(
                    target.label,
                    default_interval,
                    label_intervals,
                )
                no_click_since = time.monotonic()
                no_click_screenshot_saved = False
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
            paused, resumed = _process_keyboard_events(
                keyboard_events,
                capture,
                region,
                no_click_screenshot_dir,
                paused,
                wait_timeout=interval,
            )
            if resumed:
                no_click_since = time.monotonic()
    except pyautogui.FailSafeException:
        print("\nPyAutoGUI fail-safe activated; SmartClick stopped.")
    except KeyboardInterrupt:
        print("\nSmartClick stopped.")
    finally:
        try:
            if "keyboard_listener" in locals():
                keyboard_listener.stop()
        finally:
            capture.close()


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

import yaml
from yaml.nodes import MappingNode

from app import load_config


def _selected_region(
    monitor: dict[str, int],
    selection: tuple[int, int, int, int],
) -> dict[str, int]:
    x, y, width, height = map(int, selection)
    if width <= 0 or height <= 0:
        raise ValueError("No screen region was selected")
    region = {
        "left": monitor["left"] + x,
        "top": monitor["top"] + y,
        "width": width,
        "height": height,
    }
    if region["left"] < 0 or region["top"] < 0:
        raise ValueError("Selected region has negative screen coordinates")
    return region


def _replace_region_values(config_text: str, region: dict[str, int]) -> str:
    document = yaml.compose(config_text)
    if not isinstance(document, MappingNode):
        raise ValueError("Configuration must contain a YAML mapping")

    region_node: MappingNode | None = None
    for key_node, value_node in document.value:
        if key_node.value == "region" and isinstance(value_node, MappingNode):
            region_node = value_node
            break
    if region_node is None:
        raise ValueError("Configuration must define region as a mapping")

    replacements: list[tuple[int, int, str]] = []
    found: set[str] = set()
    for key_node, value_node in region_node.value:
        key = key_node.value
        if key in region and key not in found:
            replacements.append(
                (value_node.start_mark.index, value_node.end_mark.index, str(region[key]))
            )
            found.add(key)
    missing = set(region) - found
    if missing:
        raise ValueError(
            f"Configuration region is missing keys: {', '.join(sorted(missing))}"
        )

    for start, end, replacement in sorted(replacements, reverse=True):
        config_text = config_text[:start] + replacement + config_text[end:]
    return config_text


def save_region(config_path: Path, region: dict[str, int]) -> None:
    config_text = config_path.read_text(encoding="utf-8")
    updated_text = _replace_region_values(config_text, region)
    config_path.write_text(updated_text, encoding="utf-8")
    load_config(str(config_path))


def calibrate(config_path: Path) -> dict[str, int] | None:
    load_config(str(config_path))

    import cv2
    import mss
    import numpy as np

    with mss.mss() as screen:
        if len(screen.monitors) < 2:
            raise RuntimeError("No display is available for screen capture")
        monitor = screen.monitors[1]
        screenshot = screen.grab(monitor)
        image = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_BGRA2BGR)

    try:
        selection = cv2.selectROI(
            "Drag around the mirrored iPhone, then press Enter (Esc to cancel)",
            image,
            showCrosshair=True,
            fromCenter=False,
        )
    finally:
        cv2.destroyAllWindows()

    x, y, width, height = map(int, selection)
    if width <= 0 or height <= 0:
        print("Region selection cancelled; configuration was not changed.")
        return None

    region = _selected_region(monitor, (x, y, width, height))
    save_region(config_path, region)
    print(f"Saved screen region to {config_path}: {region}")
    return region


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select the mirrored iPhone region and save it to the app config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Application config to update (default: config/config.yaml)",
    )
    args = parser.parse_args(argv)

    try:
        calibrate(args.config)
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as error:
        print(f"Region calibration failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

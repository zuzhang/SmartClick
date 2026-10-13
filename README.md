# SmartClick

SmartClick is a macOS automation MVP for interacting with an iPhone displayed
through iPhone Mirroring. A configurable screen region is captured, a selected
detector produces normalized business labels, the decision engine chooses an
action, and the clicker performs the action only inside that region.

```text
ScreenCapture -> Detector (OCR or YOLO) -> DecisionEngine -> Clicker
```

The detector contract is defined in `detector/base.py`. Detections use
frame-relative pixel coordinates, so the decision engine and clicker do not
depend on OCR, YOLO, or other detection implementation details.

## Setup

Use Python 3.10 or newer on macOS. Install the base requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `config/config.example.yaml` to `config/config.yaml`. Set `region` to the
screen coordinates and dimensions of the mirrored iPhone window. Keep the
region tightly around the phone; clicks outside it are rejected. Coordinates
are physical screen pixels, with the origin at the top-left of the display.
For Retina displays, check that the coordinates reported by your screen
measurement tool use the same pixel scale as MSS.

Grant your terminal or Python runtime **Screen Recording** and **Accessibility**
permissions in macOS System Settings > Privacy & Security. Open and position
iPhone Mirroring before starting the bot.

## Run

```bash
python app.py --config config/config.yaml
```

The sample config defaults to OCR. EasyOCR may download its recognition models
on first startup. The bot prints each selected normalized label and click
coordinate. Press `Ctrl+C` to stop; PyAutoGUI's corner fail-safe is also
enabled. The bot does not automatically locate the phone window.

## Detector selection

All detectors implement `detect(frame) -> list[Detection]`. Labels are
normalized to `close_button`, `skip`, `confirm`, `claim`, `collect`, `continue`,
`next`, `reward`, and `play`; the decision engine never receives raw OCR text.

To use YOLO, install the optional dependency and supply a trained model whose
class names are these normalized labels:

```bash
python -m pip install ultralytics
```

Then set `detector: yolo` and `model_path: models/best.pt` in the config. YOLO
classes that are not supported labels are ignored. No trained model is
included, so the model path must point to your own weights.

## Development

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

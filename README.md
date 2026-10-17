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

If EasyOCR fails to download its models with
`CERTIFICATE_VERIFY_FAILED` and Python was installed from python.org, run the
certificate installer included with that Python version, for example:

```bash
"/Applications/Python 3.12/Install Certificates.command"
```

Use the matching version if you installed a different Python release. On
managed networks that inspect HTTPS traffic, Python may also need the
organization's trusted CA certificate.

## Run

```bash
python app.py --config config/config.yaml
```

The sample config uses YOLO on Apple Silicon via MPS and expects a trained model
at `models/best.pt`. Install Ultralytics as described in
[YOLO_TRAINING.md](YOLO_TRAINING.md). To use OCR instead, set `detector: ocr`
and uncomment the OCR-only settings in `config/config.yaml`; EasyOCR may
download its recognition models on first startup. The bot prints each selected
normalized label and click coordinate. Press `Ctrl+C` to stop; PyAutoGUI's
corner fail-safe is also enabled: move the pointer to the screen's upper-left
corner to stop, including when the terminal is not focused. When multiple
controls share the same priority, SmartClick clicks the highest-confidence
detection first, then captures a fresh frame before choosing the next action.
This lets it detect a close button revealed after dismissing an overlay. The
bot does not automatically locate the phone window.

## Detector selection

All detectors implement `detect(frame) -> list[Detection]`. Labels are
normalized to `close_button`, `skip`, `confirm`, `claim`, `collect`, `continue`,
`next`, `reward`, `play`, and `watch`; the decision engine never receives raw
OCR text.

To recognize visual controls such as an X icon, train a YOLO model on screenshots
from your game. See [YOLO_TRAINING.md](YOLO_TRAINING.md) for CVAT export,
dataset preparation, training, testing, and SmartClick configuration.

Use a labeling tool that exports Ultralytics YOLO detection labels. Put the
screenshots and matching `.txt` label files in this layout; each label file
contains one normalized `class_id x_center y_center width height` row per box:

```text
datasets/game-controls/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Label X buttons as `close_button`, confirmation buttons as `confirm`, Watch
buttons as `watch`, and GO buttons/icons as `play`. Use varied screenshots
showing the controls at different positions, sizes, and game states, and include
screenshots without target buttons. Keep corresponding image and label filenames
the same, for example `screen01.png` and `screen01.txt`.

After training, copy the best weights into `models/best.pt`, then set these
values in `config/config.yaml`:

```yaml
detector: yolo
model_path: models/best.pt
```

Use `interval_seconds` as the initial delay between detection cycles, and
`interval_seconds_by_label` to override the delay after clicking a particular
label. The current interval remains unchanged when no click occurs. For example,
the sample config uses a 1-second delay after `close_button` or `skip` and
3 seconds after `play`.

Set `no_click_screenshot_after_seconds` to save a review screenshot after that
many seconds without a successful click. The default is 60 seconds, and
`no_click_screenshot_dir` sets the output folder (default:
`review_screenshots`). Only one screenshot is saved per no-click period; a
successful click resets the timer and allows another screenshot after the next
timeout. Screenshots contain the configured capture region and are saved as PNG.

The model's class names must match the normalized labels in this project
(including spelling and underscores); unsupported classes are ignored. No
trained weights are included. Start with a conservative `min_confidence` and
test the detection output before allowing the bot to click.

## Development

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

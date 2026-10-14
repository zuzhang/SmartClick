# Training and testing a YOLO detector

This guide covers training an Ultralytics YOLO detection model on screenshots
annotated in CVAT, testing it on a new screenshot, and using the trained model
with SmartClick. Run commands from the SmartClick project root unless noted.

## 1. Prepare the Python environment

Activate the project's virtual environment. For the `.venv312` environment on
macOS or Linux:

```bash
source .venv312/bin/activate
```

If you have not created it yet:

```bash
python3 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r requirements.txt
```

Install Ultralytics into the activated environment:

```bash
python -m pip install ultralytics
python -m pip check
```

Use `python` after activating the virtual environment so training and the
application use the same installed packages. The project requirements pin
NumPy below 2; if installation reports a `contourpy` conflict because the
installed version requires NumPy 2, install a compatible version with
`python -m pip install "contourpy<1.4"` and run `python -m pip check` again.

## 2. Export and arrange the CVAT dataset

In CVAT, export using **Ultralytics YOLO Detection 1.0** and enable **Save
images**. Detection exports bounding boxes (rectangles), not ellipses. Each
annotation row has this format:

```text
class_id x_center y_center width height
```

The coordinates are normalized to the image dimensions, from 0 to 1. An image
with multiple boxes has multiple rows. The image and label basenames must match,
for example `screen01.png` and `screen01.txt`.

Arrange the exported images and labels like this:

```text
datasets/game-controls/
  images/
    train/
    val/
  labels/
    train/
    val/
```

CVAT may export all images into a single subset. In that case, split the
images between `train` and `val` yourself and move each image's matching label
file into the corresponding labels folder. For 22 images, a starting split
could be 18 train and 4 val. Keep near-duplicate screenshots together in one
split. Images without target objects can be included as negative examples; if
CVAT did not export an empty `.txt` file for one, the image can still stay in
the dataset without one.

## 3. Create the dataset configuration

Create `training/game-controls.yaml` (and the `training` directory if needed).
Set `path` to the absolute path of your dataset directory. For example, if CVAT's exported class list contains `close_button`, `skip`, and
`play` in that order:

```yaml
path: /Users/your-name/Projects/SmartClick/datasets/game-controls
train: images/train
val: images/val

names:
  0: close_button
  1: skip
  2: play
```

Preserve the complete class list and exact class-ID order from CVAT's exported
`data.yaml` or class list, even if some classes do not yet have examples. The
IDs in the `.txt` files must match this order. The numbers in SmartClick's
decision-priority mapping are not YOLO class IDs.

Before training, confirm the configured train and validation folders exist and
contain images. If you only have training data, create a validation split
first; do not point `val` at a nonexistent directory.

## 4. Train the detector

From the project root, with the virtual environment active, run:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=training/game-controls.yaml \
  epochs=100 \
  imgsz=640
```

Ultralytics downloads the pretrained `yolo11n.pt` checkpoint if necessary.
Training results are written to a run directory such as
`runs/detect/train/`; if that name already exists, Ultralytics creates another
such as `train2` or `train3`. Use the run directory printed at the end of
training. The best checkpoint is at:

```text
runs/detect/<your-run-name>/weights/best.pt
```

## 5. Test on a screenshot

Use a screenshot that was not used for training. Replace the example path with
the full path to your image:

```bash
yolo detect predict \
  model=runs/detect/<your-run-name>/weights/best.pt \
  source=/path/to/unseen-screenshot.png \
  save=True
```

The prediction output is saved in a new folder under `runs/detect/predict/`.
Open the saved image and check that boxes and class names are correct. The
prediction output also reports detections and confidence scores. If an expected
object is missing, first verify the object was labeled, the class order is
correct, and the screenshot resembles the training examples. You can lower
`conf` temporarily to inspect weaker predictions, for example `conf=0.1`, but
this can also show false positives.

You can evaluate against the validation split with:

```bash
yolo detect val \
  model=runs/detect/<your-run-name>/weights/best.pt \
  data=training/game-controls.yaml
```

With a small dataset, validation metrics can vary substantially. Test on
several varied screenshots that were not used for training as well.

## 6. Use the model with SmartClick

Copy the trained checkpoint into the project and configure the app:

```bash
mkdir -p models
cp runs/detect/<your-run-name>/weights/best.pt models/best.pt
cp config/config.example.yaml config/config.yaml
```

In `config/config.yaml`, set:

```yaml
detector: yolo
model_path: models/best.pt
```

Keep the configured `min_confidence` high enough to avoid acting on uncertain
detections. Start SmartClick and verify its detections and chosen actions before
letting it click during gameplay:

```bash
python app.py --config config/config.yaml
```

The model's class names must match SmartClick's supported labels exactly, such
as `close_button`, `skip`, and `play`. Unsupported classes are ignored.

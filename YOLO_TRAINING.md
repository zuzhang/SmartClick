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

Import the CVAT ZIP directly from the project root:

```bash
python training/import_cvat.py /path/to/cvat-export.zip
```

You can also pass the extracted export folder instead of a ZIP. The importer
copies images and matching labels into the split directories configured in
`training/game-controls.yaml`. It recognizes `train`, `Train`, `val`, `valid`,
and `Validation` regardless of case, so CVAT's lowercase `train` directory is
imported into the configured `Train` directory without editing the YAML. The
importer does not replace existing files unless you explicitly pass
`--overwrite`; filename collisions otherwise stop the import.

The export must contain both training and validation subsets. If CVAT exports
everything into one subset, first split the images and their matching labels
into train and validation folders. Keep near-duplicate screenshots together in
one split. Images without target objects can be included as negative examples;
if CVAT did not export an empty `.txt` file for one, the image can still stay
in the dataset without one.

## 3. Create the dataset configuration

The repository's `training/game-controls.yaml` is configured for the layout
above and uses `training` as the dataset root (when running commands from the
project root):

```yaml
path: training
train: images/Train
val: images/Validation

names:
  0: close_button
  1: play
  2: skip
```

You can instead use lowercase subset directories such as `images/train`,
`images/val`, `labels/train`, and `labels/val`, or set `path` to another dataset
root. In either case, make the corresponding paths and class names match your
actual data. `train` and `val` must be present in every dataset YAML: each can
refer either to a directory of images or to a text file listing image paths.

Preserve the complete class list and exact class-ID order from CVAT's exported
`data.yaml` or class list, even if some classes do not yet have examples. The
IDs in the `.txt` label files must match this order. Do not renumber existing
classes when adding images or classes. The numbers in SmartClick's
decision-priority mapping are not YOLO class IDs.

Before training, confirm the configured train and validation folders exist and
contain images (or that each configured image-list text file exists and lists
valid image paths). If you only have training data, create a validation split
first; do not point `val` at a nonexistent directory.

For example, with `path: training`, a list-based configuration can use
`train: Train.txt` and `val: Validation.txt`. Put one image path per line in
each list, relative to the dataset root, such as `images/Train/new-screen.png`.
The list entries refer to images, not label files; YOLO finds labels from the
matching image and label directory structure.

## 4. Train the detector

From the project root, with the virtual environment active, choose the value
for your hardware from this table and use it for `device` in the command below:

| Hardware | `device` value | Notes |
| --- | --- | --- |
| Apple Silicon (M1/M2/M3/M4 and later) | `mps` | Requires PyTorch MPS support. |
| NVIDIA GPU | `0` (or GPU index) | Requires CUDA-enabled PyTorch. |
| AMD GPU on Linux | `0` | Requires a compatible PyTorch ROCm setup. |
| Intel Mac, no supported GPU, or GPU fallback | `cpu` | Slower, but needs no GPU setup. |

```bash
yolo detect train \
  model=yolo11n.pt \
  data=training/game-controls.yaml \
  epochs=20 \
  imgsz=640 \
  device=<device>
```

For multiple NVIDIA GPUs, use the desired GPU index (for example, `1`). If MPS
is unavailable or unsupported by an operation, use `cpu`. Ultralytics downloads
the pretrained `yolo11n.pt` checkpoint if necessary.

Training results are written to a run directory such as
`runs/detect/train/`; if that name already exists, Ultralytics creates another
such as `train2` or `train3`. Use the run directory printed at the end of
training. The best checkpoint is at:

```text
runs/detect/<your-run-name>/weights/best.pt
```

## 5. Add more images and train again

For every new screenshot, create accurate bounding-box labels for all target
controls in the same YOLO format and class-ID order. Put each image and its
matching label file in the same subset structure, for example:

```text
training/images/Train/new-screen.png
training/labels/Train/new-screen.txt
```

Use `Validation` for images held out to measure performance. Avoid putting
near-duplicate frames or screenshots from the same short gameplay sequence in
both subsets, since that can make validation results look better than
generalization really is. With the current directory-based YAML, adding files
to those directories does not require changing the YAML. If `train` or `val`
points to a text file instead, add each new image's path to the appropriate
list file as well.

Retrain using the updated dataset and the previous run's best weights as the
starting checkpoint:

```bash
yolo detect train \
  model=runs/detect/<previous-run-name>/weights/best.pt \
  data=training/game-controls.yaml \
  epochs=20 \
  imgsz=640 \
  device=<device>
```

Use the same `device` value from the table above.

Keep the earlier training images in the training set when possible; training
only on new images can make the model forget controls it previously learned.
The command above starts a new fine-tuning run from `best.pt`. It does not
resume the previous run's optimizer state. Use `resume=True` only to continue
an interrupted run with its `last.pt`, not as the normal way to add data.

## 6. Generate a CVAT review dataset from screenshots

The helper script runs the model on every supported image in
`review_screenshots/` and creates a CVAT Ultralytics YOLO Detection dataset ZIP.
It uses `models/best.pt` and a confidence threshold of `0.2` by default. Install
Ultralytics if it is not already available in the active environment:

```bash
python -m pip install ultralytics
```

Copy the example config if you want to change the screenshot directory, model,
confidence threshold, or output directory:

```bash
cp config/auto_label.example.yaml config/auto_label.yaml
```

Edit `config/auto_label.yaml` and run:

```bash
python training/prepare_cvat_predictions.py
```

All supported image files in the configured screenshot directory, including
files in subdirectories, are processed each time. The output ZIP is written
under `cvat_imports/` by default. In CVAT, open a project, choose **Actions >
Import dataset**, select **Ultralytics YOLO Detection 1.0**, and upload the ZIP.
Importing a dataset into a project creates a new task with the images and
predicted annotations so they can be reviewed and corrected in CVAT. The ZIP
contains the original screenshots and prediction labels, not rendered preview
images. Model class names and IDs are taken directly from the model.

## 7. Test on a screenshot

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

## 8. Use the model with SmartClick

Copy the trained checkpoint into the project and configure the app:

```bash
mkdir -p models
cp runs/detect/<your-run-name>/weights/best.pt models/best.pt
cp config/config.example.yaml config/config.yaml
```

In `config/config.yaml`, set `device` to match the hardware:

```yaml
detector: yolo
model_path: models/best.pt
device: mps
```

Use `device: mps` on Apple Silicon, `device: 0` for a configured NVIDIA GPU,
and `device: cpu` to force CPU inference. You can also omit `device` to let
Ultralytics choose. The active PyTorch installation must support MPS; verify it
from the same virtual environment with:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

This should print `True`. If it prints `False`, install a PyTorch build with
Apple Silicon MPS support in that environment. The `device` setting applies to
YOLO only; it does not enable GPU use in EasyOCR.

Keep the configured `min_confidence` high enough to avoid acting on uncertain
detections. Start SmartClick and verify its detections and chosen actions before
letting it click during gameplay:

```bash
python app.py --config config/config.yaml
```

For each recognized control, SmartClick prints its screen coordinate (the
coordinate sent to the mouse), its position relative to the captured region,
and its confidence. It clicks at the center of the YOLO bounding box or OCR
text box. If detections are printed but none meet `min_confidence`, the app
reports that it will not click; if nothing is recognized, it reports that too.

The model's class names must match SmartClick's supported labels exactly, such
as `close_button`, `skip`, and `play`. Unsupported classes are ignored.

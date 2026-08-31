# Book Spine Detector — ML Training

This repository contains the first training scaffold for a real-time mobile book counter.
The model segments the long visible boundaries of each book. Post-processing projects those
boundaries onto one axis, locates peaks, and estimates the number of books as
`unique boundaries - 1`.

The current confidence score and counting post-processing are experimental. They are intended
for public-dataset feasibility work, not manufacturing QC acceptance.

## What is included

- A mobile-oriented LR-ASPP model with a MobileNetV3-Large backbone.
- A converter for YOLO detection boxes, oriented boxes, and segmentation polygons.
- Weighted cross-entropy plus Dice training loss for thin separator masks.
- Train, evaluate, and ONNX export command-line tools.
- A Colab/local Jupyter training notebook.
- A Colab-native upload workflow for manually testing photographs.

## 1. Create an environment

Python 3.10–3.12 is the most conservative choice for hosted GPU environments.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,dev]"
```

For Google Colab, choose **Runtime → Change runtime type → GPU**, clone the repository under
`/content/book-spine-detector`, then work from `/content/book-spine-detector/ml` and open
`ml/notebooks/train_colab.ipynb`.

## 2. Obtain a public dataset

The recommended first candidate is the Roboflow
[spine-seg dataset](https://universe.roboflow.com/dobs-tnsmf/spine-seg-3eyao), which declares
CC BY 4.0 and contains instance-segmentation annotations. Export a non-augmented version as
YOLOv8 segmentation if possible.

Keep the dataset license and attribution next to any downloaded data. Public community dataset
licensing and image provenance still require review before commercial deployment.

Expected source structure:

```text
data/raw/public-book-spines/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 3. Convert annotations to separator masks

```bash
book-spine-prepare-yolo \
  --source data/raw/public-book-spines \
  --destination data/processed/public_books \
  --line-width 5 \
  --prefix spineseg_
```

For every annotated polygon or box, the converter estimates an oriented rectangle with PCA and
draws its two long edges. `--line-width` is expressed at a 640-pixel model scale and is adjusted
for the source image resolution. Inspect the results in the training notebook before training. Public
annotations can be noisy, and a few incorrect masks can materially affect thin-line learning.

When adding a second public source, use a different `--prefix` to avoid filename collisions.

## 4. Train

Edit `configs/baseline.yaml`, especially the dataset path and batch size, then run:

```bash
book-spine-train --config configs/baseline.yaml
```

Resume an interrupted Colab run with:

```bash
book-spine-train \
  --config configs/baseline.yaml \
  --resume runs/separator-baseline/last.pt
```

Generated files:

```text
runs/separator-baseline/
├── best.pt
├── last.pt
├── history.json
└── resolved_config.json
```

The pretrained backbone is downloaded the first time it is used. Set
`model.pretrained_backbone: false` only for a fully offline run.

If the GPU runs out of memory, reduce the batch size to `4` or the image size to `512`. Prefer
reducing batch size first because resizing can erase thin separators.

## 5. Evaluate

```bash
book-spine-evaluate \
  --checkpoint runs/separator-baseline/best.pt \
  --split test
```

This reports segmentation Dice/IoU and provisional count MAE/exact-count accuracy. Public
bookshelf metrics do not represent expected factory accuracy.

## 6. Test manually in Google Colab

Use the native upload cell at the end of `notebooks/train_colab.ipynb`. It loads one image with
`google.colab.files.upload()`, runs the selected checkpoint, and displays the detected boundaries
and count. Change the post-processing values in that cell to tune orientation, pixel threshold,
peak height, peak prominence, and minimum boundary distance.

Increasing minimum distance merges nearby double detections. Increasing prominence or peak
height removes weak false separators. The native Colab uploader is intentionally kept in the
notebook rather than the reusable Python package.

## 7. Export ONNX

```bash
book-spine-export \
  --checkpoint runs/separator-baseline/best.pt \
  --output runs/separator-baseline/separator_model.onnx
```

Before embedding the model in Android or iOS, compare PyTorch and ONNX outputs on a fixed set of
golden images. Quantization should happen only after the floating-point model is accurate.

## Project layout

```text
ml/
├── configs/
├── data/
├── notebooks/
│   └── train_colab.ipynb
├── src/book_spine_ml/
├── tests/
└── pyproject.toml
```

## Current limitations

- The public datasets predominantly contain vertical library shelves, not factory stacks.
- Count confidence is a peak-quality heuristic and is not calibrated.
- Video stabilization and high-resolution verification are not implemented yet.
- The converter approximates irregular polygons using an oriented rectangle.
- A production model still needs labeled factory images and a factory-only test set.

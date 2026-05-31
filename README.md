# PyClouds

PyClouds is an open-source cloud detection and cloud coverage estimation framework written in Python.

It is in early development stage for now.

The project aims to provide a lightweight, extensible, and community-driven solution for:

* estimating cloud coverage from all-sky images,
* creating and editing cloud masks,
* training custom machine learning models,
* sharing training datasets to continuously improve a common cloud detection model

PyClouds was originally designed for astronomy applications (all-sky cameras, observatories, weather
assessment), but it can be adapted to many other cloud detection use cases.

---

# Features

## Cloud Identification

PyClouds can estimate cloud coverage on a single image using a trained segmentation model.

Features:

* Cloud coverage percentage estimation
* Editable cloud masks
* Optional valid-sky mask support
* CPU or GPU execution
* Cloud overlay visualization
* Command line and graphical interfaces

---

## Cloud Mask Editor

The graphical editor allows manual correction of automatically generated masks.

Available tools:

* Brush
* Magic selection
* Closed-area fill
* Undo
* Reset
* Overlay visualization
* Zoom support

This allows users to rapidly create high-quality image/mask training pairs.

---

## Model Training

PyClouds includes a graphical training interface allowing users to train new models from their own datasets.

Features:

* U-Net based segmentation
* Multiple encoders
* Automatic train/validation split
* Early stopping
* Threshold optimization
* CUDA acceleration

The resulting model can be immediately reused by the identification tools.

---

# Installation

## Create a virtual environment

Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```cmd
python -m venv venv
venv\Scripts\activate
```

---

## Install CPU version

```bash
python install-requirements.py --mode cpu
```

---

## Install GPU version

Example:

```bash
python install-requirements.py --mode gpu126
```
---

# Cloud Identification

## Graphical Interface

Launch:

```bash
PYTHONPATH=. ./bin/cloud-identifier-ui.py
```

Or directly open an image:

```bash
PYTHONPATH=. ./bin/cloud-identifier-ui.py image.jpg
```

The UI allows:

* loading cloudy images,
* running cloud detection,
* editing masks,
* saving masks,
* creating training pairs for future model training.

Notice you can also use `pip install -e .` once if you want to launch `./bin/cloud-identifier-ui.py` and
other scripts directly without `PYTHONPATH=. ...` 
---

## Command Line Interface

A command-line version is also available:

```bash
PYTHONPATH=. ./bin/cloud-identifier.py \
    --input image.jpg \
    --out overlay.jpg
```

For scripting, automation, or batch processing.

---

# Creating Training Data

One of the main goals of PyClouds is to simplify the creation of training datasets.

Typical workflow:

1. Load an image.
2. Run cloud identification.
3. Correct the mask manually.
4. Save the image/mask pair.

The resulting files typically look like:

```text
2026-05-13-20-25-03.jpg
2026-05-13-20-25-03_mask.png
```

These pairs can be used directly for training and will likely be stored in a dedicated directory.

---

# Training a Model

Training can be performed either on CPU or GPU.

GPU training is strongly recommended for large datasets or frequent model retraining.

Launch:

```bash
PYTHONPATH=. ./bin/cloud-train-ui.py
```

Select:

* training directory,
* output model,
* encoder,
* learning parameters,

and start training.

The generated `.pth` model can immediately be reused by:

```bash
cloud-identifier.py
```

or

```bash
cloud-identifier-ui.py
```

using:

```bash
--model my_model.pth
```

---

# Community Model

Finally, PyClouds is intended to be a collaborative project.

If you create image/mask pairs, you are encouraged to share them with the project.

Contributed training pairs can be merged into a common dataset used to train improved public models.

Benefits:

* Better cloud detection
* More robust models
* Wider weather conditions
* Different cameras and optics
* Improved generalization

Every contributor benefits from improvements made possible by the community dataset.

---

# Recommended Data

Useful training data includes:

* All-sky cameras
* Observatory cameras
* Fisheye lenses
* Daytime images
* Nighttime images
* Partial cloud cover
* Overcast conditions
* Thin clouds
* High clouds
* Contrails
* Moonlit clouds
* Different seasons
* Different climates
* Different geographic locations
* Different moon phases

Diversity is valuable...

---

# License

See LICENSE file.

---

# Contributing

Contributions are welcome.

Bug reports, feature requests, training datasets, documentation improvements, and pull requests are greatly appreciated.


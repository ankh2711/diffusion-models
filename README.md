# Automatic Image Restoration System

## Overview

This project implements an automatic image restoration system based on diffusion models and neural network classifiers.
The system analyzes the input image, determines the type of distortion and dataset domain, automatically generates a mask, selects the most suitable restoration model, and performs image inpainting.

The project combines several modern diffusion-based restoration approaches:

* RePaint
* DiffPIR
* CoPaint

Additionally, the system uses EfficientNet-based classifiers to:

* classify image distortions,
* classify image domains/datasets,
* select the optimal restoration model depending on quality or speed preference.

---

# Supported Distortion Types

The system supports the following distortion classes:

| Label | Distortion Type        |
| ----- | ---------------------- |
| 0     | Periodic Noise         |
| 1     | Grid                   |
| 2     | Stripes                |
| 3     | Gaussian Noise         |
| 4     | Half Missing           |
| 5     | Rectangle Missing Area |
| 6     | Border Damage          |

---

# Supported Dataset Types

| Label | Dataset  |
| ----- | -------- |
| 0     | Celeba   |
| 1     | Imagenet |
| 2     | Places2  |

---

# Project Structure

```text
project/
│
├── RePaint/
├── DiffPIR/
├── CoPaint/
├── distortion_classifier.pth
├── data_type_classifier.pth
│
├── main.py
└── README.md
```

---

# Requirements

Install dependencies before running the project.

## Python Version

```text
Python 3.9+
```

## Required Libraries

```bash
pip install torch torchvision pillow numpy opencv-python
```

Additional dependencies are required for:

* RePaint
* DiffPIR
* CoPaint

Please install dependencies for each framework separately according to their official repositories.

---

# Repository Setup

Before running the project, download the required restoration frameworks and place them in the project root directory:

```bash
git clone https://github.com/andreas128/RePaint.git
git clone https://github.com/ucsb-nlp-chang/copaint.git CoPaint
git clone https://github.com/yuanzhi-zhu/DiffPIR.git
```

After downloading the repositories, replace several original files with the modified versions provided in this project.

## Required File Replacement

Replace the following files inside the downloaded repositories:

| Repository | Original File           |
| ---------- | ----------------------- |
| CoPaint    | `CoPaint/main.py`       |
| DiffPIR    | `DiffPIR/main_ddpir.py` |
| RePaint    | `RePaint/test.py`       |

The modified versions of these files are included in the root directory of this project:

```text
main_copaint.py
main_diffpir.py
test.py
```

Copy them as follows:

```text
main_copaint.py  →  CoPaint/main.py

main_diffpir.py  →  DiffPIR/main_ddpir.py

test.py          →  RePaint/test.py
```

These modified files contain the changes required for integration with the automatic restoration pipeline, including automatic mask processing, model invocation, and result handling.

## Model Checkpoints

Download the classifier checkpoints and place them in the project root directory:

* Distortion classifier (`distortion_classifier.pth`):
  [Download distortion_classifier.pth](https://disk.yandex.ru/d/LcfwCNRhtNcBGg)

* Data type classifier (`data_type_classifier.pth`):
  [Download data_type_classifier.pth](https://disk.yandex.ru/d/TSNRzGIJiRbqVQ)

---

# Running the Project

Run:

```bash
python main.py
```

A graphical window will appear:

1. Select an image
2. Choose restoration preference:

   * `1` — quality
   * `2` — speed

The restored image will be saved as:

```text
restored_result.png
```

Generated masks are stored in:

```text
generated_masks/
```

---

# Example Workflow

```text
Input image
    ↓
Distortion classification
    ↓
Mask generation
    ↓
Dataset classification
    ↓
Automatic model selection
    ↓
Image restoration
    ↓
Restored image
```

---

# Main Functions

## `load_classifier()`

Loads the distortion classification model.

## `load_dataset_classifier()`

Loads the dataset/domain classifier.

## `generate_mask()`

Generates a binary mask depending on distortion type.

## `choose_model()`

Selects the optimal restoration model.

## `auto_inpaint()`

Main restoration pipeline:

* classification,
* mask generation,
* model selection,
* inpainting.

---

# Author

Ani Khechoyan
Applied Mathematics and Informatics
Moscow State University

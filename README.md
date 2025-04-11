# 🖼️ Neural Style Transfer with PyTorch

This project implements a Neural Style Transfer algorithm using `PyTorch`. It blends the content of one image with the style of another, generating a unique, stylized result.

## Table of Contents

1. [Features](#features)
2. [Setup and Installation](#setup-and-installation)
3. [Inference](#inference)
4. [Results](#results)

## Features

- Built using `PyTorch` and `Torchvision`
- Uses a pretrained `VGG19` for feature extraction
- Gram matrix loss for style transfer
- Command-line interface for customizing input and training

## Setup And Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/matin-ghorbani/Neural-Style-Transfer
    cd Neural-Style-Transfer
    ```

2. **Install the Required Packages:**
    Ensure you have Python 3.9+ and PyTorch installed. Install the dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Inference

Run this command:

```bash
python3 main.py --org-img images/matin.png \
    --style-img images/starry_night_style.jpg
```

You can also see the other arguments of it with this command

```bash
clear; python main.py --help
```

*For Example:*

- *`--steps`*: Total steps to modify the original image. **default:***`6000`*
- *`--save-samples, --no-save-samples`*: Save sample images or not. **default:***`True`*

## Results

| **Starry Night** | **Edvard Munch** | **Water Lilies** |
|:----------------:|:----------------:|:----------------:|
| <img src="./images/starry_night_style.jpg" width="350" alt="starry night output"> | <img src="./images/edvard_munch_style.jpg" width="350" alt="edvar munch output"> | <img src="./images/Water_Lilies.jpg" width="350" alt="water lilies output"> |

| **Output** | **Output** | **Output** |
|:----------------:|:----------------:|:----------------:|
| ![Starry Night Output](./images/outputs/starry_night_generated.gif) | ![Edvar Munch Output](./images/outputs/edvard_munch_style.gif) | ![Water Lilies Output](./images/outputs/water_lilies_style.gif) |

The main image was:
---
<img src="./images/matin.png" width="300" alt="main image">

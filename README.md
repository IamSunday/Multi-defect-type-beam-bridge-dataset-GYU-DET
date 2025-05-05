# Multi-defect-type-beam-bridge-dataset-GYU-DET

This repository provides training code and label conversion tools for the [GYU-DET dataset](https://doi.org/10.57760/sciencedb.19893), a multi-defect-type beam bridge image dataset designed for object detection tasks.

## Repository Structure

- `ultralytics/`: Contains YOLOv11 training and configuration code.
- `txt2coco.py`: Converts YOLO-format labels into COCO-style `.json` annotations.
- `txt2xml.py`: Converts YOLO-format labels into VOC-style `.xml` annotations.

## Installation

Clone this repository:

```bash
git clone https://github.com/IamSunday/Multi-defect-type-beam-bridge-dataset-GYU-DET.git
```

## Dataset

Download the GYU-DET dataset from:

https://doi.org/10.57760/sciencedb.19893

## Quick Start
After downloading the dataset, edit the ultralytics/GYU-DET.yaml file to point to the correct paths for train, val, and test datasets.

Run the training script:

```
python ultralytics/mytrain.py
```

## Dependencies

Ensure the following dependencies are installed:

Python 3.11

torch == 2.3.1

CUDA == 12.1

torchvision == 0.18.1

ultralytics == 8.3.33

numpy == 1.26.3

matplotlib == 3.9.0

opencv-python == 4.10.0.84

scipy == 1.13.1

tqdm == 4.65.2

ultralytics == 8.0.227

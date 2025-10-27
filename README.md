# Pothole Detection System with Classification

## Introduction

This repository contains a complete system for detecting potholes using computer vision. The project uses a **YOLO (You Only Look Once)** model for fast, real-time object detection.

To increase accuracy and reduce false positives, this system also implements a **secondary classification model**. This classifier takes the bounding box (the "patch") detected by YOLO and performs a second check to confirm if the detected object is *actually* a pothole. This two-step process adds a valuable layer of certainty to the detections.

This project includes all the necessary scripts for data preparation, augmentation, training both models, and running inference.

-----

## Project Structure

*(Add your project directory tree here. Here is a common example you can adapt):*

```
.
├── model.py            # Main script to run the pipeline
├── resize.py           # Utility for image resizing
├── rotate.py           # Utility for image augmentation
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── classifier_data/    # Data for the classification model
│   ├── pothole/
│   └── not_pothole/
└── weights/
    ├── yolo_best.pt
    └── classifier_best.pt
```

-----

## Key Scripts

### `model.py`

This is the main, menu-driven script for managing the entire project pipeline. It provides a command-line interface with 5 options to choose from.

To run it:

```bash
python model.py
```

You will be prompted to select one of the following operations:

  * **1. Augment Data:**
    This option runs the data augmentation pipeline. It uses scripts like `rotate.py` and other techniques to create new training samples from your existing data. This helps to build a more-robust model that can generalize better.

  * **2. Train Classifier:**
    This trains the secondary classification model. This model learns to distinguish between image patches that *are* potholes and patches that *are not* (e.g., shadows, cracks, or other false positives from YOLO).

  * **3. Add Annotation for YOLO Model:**
    This option processes your dataset and prepares the annotations (bounding boxes) into the specific format required by the YOLO model for training.

  * **4. Train YOLO Model:**
    This begins the training process for the main YOLO object detection model. It uses the prepared dataset and annotations to learn how to find and draw bounding boxes around potholes.

  * **5. Start Inference:**
    This runs the final, trained system. It will load both the YOLO model and the classifier. When running on an image or video, YOLO will first detect potential potholes, and then the classifier will verify each detection to provide a final, high-certainty output.

-----

### Helper Scripts

These are smaller utility scripts used during the data preparation phase, which are likely called by **Option 1** in `model.py`.

  * **`resize.py`**
    This script is used to resize all images in the dataset to a uniform dimension. This is a crucial preprocessing step to ensure consistency and to efficiently manage computational power and memory during training.

  * **`rotate.py`**
    This is a data augmentation script. It rotates the training images by a few degrees in either direction. This creates new training data, helping the model become less sensitive to the camera's orientation and improving its overall robustness.

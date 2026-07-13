# Pothole Detection Using MobileNetV2 Image Classification

## Project Description

This is a simple image classification project for detecting whether a road image contains a pothole or shows a normal road.

The project uses MobileNetV2 with transfer learning. MobileNetV2 is loaded with ImageNet weights, and only a small classification head is added for this project.

## Dataset

Place the dataset inside the `dataset` folder.

The project supports this format:

```text
dataset/
    train/
        Pothole/
        Plain/
    valid/
        Pothole/
        Plain/
    test/
        Pothole/
        Plain/
```

The current dataset also works in this format:

```text
dataset/
    train/
        Pothole/
        Plain/
    test/
        Pothole/
        Plain/
```

It also supports this simple format:

```text
dataset/
    Pothole/
    Plain/
```

Here `Plain` means normal road images.

If only `Pothole` and `Plain` folders are present, the script automatically splits the data into:

- 80% training
- 10% validation
- 10% testing

Supported image formats are `jpg`, `jpeg`, and `png`.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Training

Run this command to train the model:

```bash
python train.py
```

The training script will:

- Load the dataset
- Resize images to 224 x 224
- Apply simple data augmentation
- Train MobileNetV2 using transfer learning
- Save the model as `pothole_mobilenetv2_model.keras`
- Generate accuracy and loss graphs
- Print accuracy, precision, recall, F1 score, confusion matrix, and classification report

## Prediction

Run this command:

```bash
python predict.py
```

Then enter the image path when asked.

Example output:

```text
Prediction : Pothole
Confidence : 97.56%
```

## Results

After training, the project displays:

- Accuracy vs Epoch graph
- Loss vs Epoch graph
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

The final result depends on the quality and size of the dataset.


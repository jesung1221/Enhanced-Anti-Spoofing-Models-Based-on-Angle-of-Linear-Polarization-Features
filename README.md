# Enhanced Anti-Spoofing Using Angle of Linear Polarization Features

This repository contains the code and resources for the research titled **"Enhanced Anti-Spoofing Models Based on Angle of Linear Polarization Features"**. The project explores a novel approach to face anti-spoofing using the **Angle of Linear Polarization (AoLP)**, leveraging the MobileNetV3 architecture to achieve efficient and robust results.

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments and Results](#experiments-and-results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Introduction
Face recognition is a critical component of biometric authentication systems. However, these systems are vulnerable to spoofing attacks, including printed photographs, screens, or masks. This project presents a robust anti-spoofing approach using AoLP, which captures polarization data at four angles (0°, 45°, 90°, 135°) and generates an AoLP colormap. This method outperforms conventional RGB-based techniques, especially under low-light conditions.

### Goals
1. Establish AoLP as a powerful anti-spoofing feature.
2. Develop a lightweight MobileNetV3-based classifier.
3. Investigate sparse AoLP data for cost-effective implementation.

---

## Key Features
- **AoLP-Based Anti-Spoofing**: Innovative use of polarization features for detecting spoof attacks.
- **MobileNetV3 Integration**: Efficient and accurate classification model for real-time applications.
- **Sparsity Investigation**: Exploration of sparsely sampled polarization pixels to reduce hardware requirements.
- **Comprehensive Metrics**: Evaluation using accuracy, EER, HTER, FAR, and FRR.

---

## Directory Structure
```plaintext
├── dataset_preparation.py      # Data preprocessing and loading
├── model_definition.py         # MobileNetV3 model definition
├── train_model_with_metrics_updated.py # Model training and evaluation
├── README.md                   # Project documentation
├── results/                    # Directory to store results and metrics
├── datasets/                   # Dataset containing real, paper, and screen images
└── iteration_weights/          # Directory for iteration-wise weights
```

## Installation
### Prerequisites
- Python 3.8 or higher
- PyTorch 1.12 or higher
- CUDA (if using GPU)

## Usage
### 1. Data Preparation
Organize your dataset as:
```plaintext
datasets/
├── train/
│   ├── face/
│   ├── paper/
│   └── screen/
├── test/
│   ├── face/
│   ├── paper/
│   └── screen/
```
### 2. Train the Model
Run the following command to train the model:
```bash
python train_model_with_metrics_updated.py
```
### 3. Evaluate the Model
The training script automatically evaluates the model on the test set after each iteration. Results are logged in the 100iterationResult_with_metrics.txt file.

## Experiments and Results
### Key Findings
1. High Accuracy: Achieved 98.97% mean test accuracy with AoLP sparsity3.
2. Robust Under Low Light: Outperformed RGB methods in illumination conditions as low as 1 lux.
3. Cost-Effective Sparsity: Maintained >97% accuracy with only 21x21 pixels (sparsity20).
Refer to the thesis for detailed results and analysis.

## Citation
If you use this repository for your research, please cite:
```plaintext
@thesis{kim2024aolp,
  author = {JaeSeong Kim},
  title = {Enhanced Anti-Spoofing Models Based on Angle of Linear Polarization Features},
  year = {2024},
  school = {Tel Aviv University}
}
```
## Acknowledgments
Special thanks to Prof. David Mendlovic and Dr. Michael Scherer for their invaluable guidance, mentorship, and support throughout this research. I am also deeply grateful to Abraham (Avi) Pelz and Guy Lifchitz from Corephotonics for their exceptional assistance and insights into neural network training algorithms and methods, which significantly contributed to the success of this work. This research was conducted at Tel Aviv University in collaboration with Corephotonics, a Samsung company.







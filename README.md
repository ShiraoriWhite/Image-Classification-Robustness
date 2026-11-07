# Image Classification Robustness using Convolutional Neural Networks

## Overview

This project was developed as part of the **Visual Image Processing and Management** university course.

The main objective is to investigate the use of Convolutional Neural Networks (CNNs) for image classification, focusing on their robustness against different types of image degradation.

The project involves designing, training, and evaluating custom CNN architectures from scratch, experimenting with different network configurations, image preprocessing techniques, and data augmentation strategies.

The repository contains the implementation of **Task 2 and Task 3** of the examination assignment.

**Note:** Task 1 is not included in this repository because it was independently developed by another member of the project group.

The original examination assignment is available in the `Documentation/` directory.

---

## Technologies and Tools

- **MATLAB** – Development environment.
- **Deep Learning Toolbox** – CNN architecture design and training.
- **Image Processing Toolbox** – Image preprocessing and augmentation.
- **Convolutional Neural Networks (CNNs)** – Image classification.
- **Data Augmentation** – Simulation of image degradation conditions.

---

## Repository Structure

```text
Image-Classification-Robustness/
│
├── README.md
│
├── Documentation/
│   └── Exam_Assignment.pdf
│
├── Task2/
│   └── Task2_CNN.m
│
├── Task3/
│   └── Task3_CNN.m
```

---

# Task 2 – Custom CNN with Data Augmentation

## Objective

The objective of Task 2 is to design and train a custom Convolutional Neural Network from scratch for image classification.

The network is trained using the original training dataset and evaluated on a degraded test dataset.

Different preprocessing and data augmentation configurations were tested to investigate their impact on classification performance.

## CNN Architecture

The network receives RGB images resized to **64 × 64 × 3**.

Its architecture consists of:

| Layer | Configuration |
|---|---|
| Input | 64 × 64 × 3 |
| Convolutional Block 1 | 64 filters, 3 × 3 |
| Batch Normalization | Yes |
| ReLU | Yes |
| Max Pooling | 2 × 2 |
| Convolutional Block 2 | 128 filters, 3 × 3 |
| Batch Normalization | Yes |
| ReLU | Yes |
| Max Pooling | 2 × 2 |
| Convolutional Block 3 | 256 filters, 3 × 3 |
| Batch Normalization | Yes |
| ReLU | Yes |
| Dropout | 0.4 |
| Fully Connected | Number of classes |
| Softmax | Classification probabilities |

The network is trained from scratch using the Adam optimizer.

### Training Configuration

- Optimizer: Adam
- Maximum epochs: 30
- Batch size: 64
- Initial learning rate: 0.001
- L2 regularization: 0.001
- Learning rate decay: Every 10 epochs
- Validation dataset: Original validation set

## Data Augmentation

During training, one random degradation is applied to each image.

The implemented techniques include:

- **JPEG Compression:** Random compression quality between 60 and 85.
- **Gaussian Blur:** Application of Gaussian filtering.
- **Gaussian Noise:** Addition of Gaussian noise.
- **Brightness Adjustment:** Random modification of image brightness.
- **Vignetting:** Simulation of artificial illumination degradation.
- **Color Desaturation:** Conversion to grayscale.
- **Horizontal Flip:** Random horizontal image flipping.

Only one of the six main degradation techniques is selected for each image, followed by a random horizontal flip.

## Test Preprocessing

The test dataset contains degraded images.

A lightweight preprocessing procedure is applied before classification, including:

- Image resizing.
- Vignetting correction.
- Partial color restoration through grayscale blending.

Different preprocessing configurations were tested to identify the most effective approach.

## Task 2 – Experimental Results

Three configurations were evaluated by combining different data augmentation and preprocessing strategies.

| Configuration | Data Augmentation | Preprocessing | Accuracy |
|---|---|---|---:|
| Strong preprocessing | Single, non-aggressive degradation | Strong | 21.6% |
| Intermediate configuration | Single, non-aggressive degradation | Light | 36.0% |
| Final configuration | Single, non-aggressive degradation | Improved light preprocessing | **37.2%** |

### Observations

The experimental results show that aggressive preprocessing negatively affected classification performance.

Reducing the intensity of preprocessing operations led to a significant improvement in accuracy.

The final configuration achieved the best result, reaching **37.2% accuracy** on the degraded test dataset.

This suggests that excessive image correction can remove useful visual information, negatively affecting the network's ability to recognize image features.

---

# Task 3 – Custom CNN Architecture Comparison

## Objective

Task 3 focuses on evaluating different custom CNN architectures under challenging image classification conditions.

Several network configurations were designed and tested to investigate the relationship between model complexity, input image resolution, and classification performance.

Unlike transfer learning approaches, all networks were trained from scratch, without using pre-trained weights.

## CNN Architectures

Three different architectures were evaluated.

| Architecture | Parameters | Layers | Input Size |
|---|---:|---:|---|
| 3 Convolutional Blocks | 9.8K | 36 | 64 × 64 × 3 |
| 4 Convolutional Blocks | 480K | 24 | 128 × 128 × 3 |
| 4 Convolutional Blocks | 415K | 21 | 128 × 128 × 3 |

The final implementation included a four-block CNN architecture with:

- Convolutional layers with increasing filter counts.
- Batch Normalization.
- ReLU activation functions.
- Max Pooling.
- Global Average Pooling.
- Dropout.
- Fully Connected classification layer.
- Softmax output.

The network receives RGB images resized to **128 × 128 × 3**.

## Data Augmentation

Task 3 introduces a more extensive augmentation strategy.

For each image, one primary degradation is randomly selected:

- **JPEG Compression:** Random compression quality between 40 and 70.
- **Gaussian Blur:** Stronger Gaussian filtering.
- **Gaussian Noise:** Addition of Gaussian noise.
- **Brightness Adjustment:** Random brightness modification.

Additionally, two optional degradations can be independently applied:

- **Vignetting:** Artificial darkening towards the image borders, applied with a 40% probability.
- **Color Desaturation:** Removal of color information, applied with a 50% probability.

A random horizontal flip is also applied.

This strategy allows the generation of images affected by multiple simultaneous degradation effects.

## Training Configuration

- Optimizer: Adam
- Maximum epochs: 150
- Batch size: 32
- Initial learning rate: 0.005
- L2 regularization: 0.00005
- Learning rate decay: Every 25 epochs
- Training approach: From scratch
- Pre-trained weights: Not used

## Task 3 – Experimental Results

The final model was evaluated on a degraded test dataset containing 100 different classes, with only 5 images available per class.

The following results were obtained:

| Metric | Result |
|---|---:|
| Top-1 Accuracy | 15.0% |
| Top-5 Accuracy | 36.4% |

### Observations

Despite the challenging experimental conditions, the model achieved a Top-1 accuracy of 15% and a Top-5 accuracy of 36.4%.

The classification task was particularly challenging due to:

- 100 different image classes.
- Only 5 test images per class.
- Significant image degradation in the test dataset.
- Training performed entirely from scratch.
- No pre-trained models or transfer learning.

The results highlight the difficulty of learning robust visual representations when training data is limited and test images present substantial quality degradation.

---

# Final Results and Conclusions

## Task 2

The degradation affecting the test images caused a significant reduction in classification performance.

To address this problem, different strategies were investigated:

- Simulating image degradations during training through data augmentation.
- Reducing the intensity of image transformations.
- Applying only one primary degradation at a time.
- Experimenting with different preprocessing configurations.

The best configuration achieved **37.2% accuracy**, demonstrating the importance of balancing data augmentation and preprocessing intensity.

## Task 3

Task 3 explored the performance of custom CNN architectures under more challenging conditions.

The final model achieved:

- **Top-1 Accuracy: 15.0%**
- **Top-5 Accuracy: 36.4%**

These results were obtained despite the following constraints:

- 100 classification classes.
- Only 5 test images per class.
- Degraded test images.
- No pre-training or transfer learning.

The experiments provided insights into the challenges of image classification under severe degradation and limited data availability.

---

# Dataset Organization

The dataset is expected to be organized into separate directories for training, validation, and testing.

Images must be organized into subfolders corresponding to their respective classes.

Example:

```text
Dataset/
│
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
├── valid/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
└── test_degradato/
    ├── class_1/
    ├── class_2/
    └── ...
```

The dataset is not included in this repository.

Make sure to update the dataset paths in the MATLAB scripts according to your local directory structure.

---

# How to Run

1. Clone or download this repository.
2. Open MATLAB.
3. Make sure the required MATLAB toolboxes are installed.
4. Place the dataset in the expected directory structure.
5. Open the desired task script.
6. Execute the script.

**Task 2:** Run `Task2_CNN.m`.

**Task 3:** Run `Task3_CNN.m`.

The scripts will perform the corresponding training and evaluation procedures.

---

## Academic Context

**Course:** Visual Image Processing and Management

**Project Type:** University Examination Project

**Main Topics:**
- Deep Learning
- Convolutional Neural Networks
- Image Classification
- Image Preprocessing
- Data Augmentation
- Image Degradation
- Model Evaluation
- Neural Network Architecture Design

The original examination assignment is included in the `Documentation/` directory.

---

## Author

Developed as part of the Visual Image Processing and Management university course.

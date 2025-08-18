# Lung Cancer Detection Models

This repository contains the machine learning and deep learning code to train models that detect lung cancer from both CT scans and histopathology images. The models are designed to classify images into categories such as malignant, benign, or normal, based on modality.

## Technologies Used

- Python 3.10+
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn
- OpenCV / PIL
- scikit-learn

## What We Do

- Preprocess and clean datasets from multiple sources
- Resize and normalize all images to 224x224
- Apply data augmentation to underrepresented classes (e.g., CT benign)
- Train individual models for:
  - CT image classification
  - Histopathology image classification
- Save trained models in `.h5` format for deployment

## Features

- Consistent data preprocessing pipeline
- Class balancing through augmentation
- Separate training for each modality
- Compatibility with backend APIs

### Contents

The repo includes **9 Jupyter Notebooks**:

#### 1. CT Models (4 notebooks)

Each of the following models was trained and evaluated independently:

- `CT_Model_ResNet50.ipynb`
- `CT_Model_EfficientNetB0.ipynb`
- `CT_Model_DenseNet121.ipynb`
- `CT_Model_MobileNetV2.ipynb`

#### 2. CT Model Comparison (1 notebook)

- `CT_Model_Comparison.ipynb`: Evaluates all four CT models using accuracy, F1-score, and AUC to select the best-performing one.

#### 3. Histopathology Models (3 notebooks)

- `Histopath_Model_ResNet50.ipynb`
- `Histopath_Model_InceptionV3.ipynb`
- `Histopath_Model_VGG16.ipynb`

#### 4. Histopathology Model Comparison (1 notebook)

- `Histopath_Model_Comparison.ipynb`: Compares all three histopathology models based on performance metrics to identify the most robust architecture.

---

### Sample Dataset for Testing

A small version of the dataset is included for quick testing and experimentation.

- Each class folder contains **20 sample images** for testing purposes.
- Follows the structure:

```
dataset/
├── ct/
│   ├── test/
│   │   ├── benign/
│   │   ├── malignant/
│   │   └── normal/
├── histopathology/
│   ├── test/
│   │   ├── adenocarcinoma/
│   │   ├── squamous_cell_carcinoma/
│   │   └── benign/
```

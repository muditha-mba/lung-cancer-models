# 🧠 Lung Cancer Detection Models

This repository contains the machine learning and deep learning code to train models that detect lung cancer from both CT scans and histopathology images. The models are designed to classify images into categories such as malignant, benign, or normal, based on modality.

## 🚀 Technologies Used
- Python 3.10+
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn
- OpenCV / PIL
- scikit-learn

## 🔍 What We Do
- Preprocess and clean datasets from multiple sources
- Resize and normalize all images to 224x224
- Apply data augmentation to underrepresented classes (e.g., CT benign)
- Train individual models for:
  - CT image classification
  - Histopathology image classification
- Save trained models in `.h5` format for deployment

## 🌟 Features
- Consistent data preprocessing pipeline
- Class balancing through augmentation
- Separate training for each modality
- Compatibility with backend APIs

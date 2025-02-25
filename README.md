# Alzheimer's Detection using Deep Learning 🧠🔬
A deep learning model for early Alzheimer’s disease detection using MRI scans. This project leverages advanced neural networks to classify different stages of Alzheimer’s, aiding in early diagnosis and intervention. Includes data preprocessing, model training, evaluation, and visualization.

The model achieves **97% accuracy** with an **AUC score of 0.9985** and an **Expected Calibration Error (ECE) of 0.0484**, ensuring both high accuracy and confidence in predictions.

## 📌 Project Overview
- Uses a CNN-based deep learning model for classification.
- Trained on medical image datasets with extensive preprocessing.
- Evaluation includes accuracy, AUC score, and calibration analysis.

## 🚀 Results
- **Test Accuracy:** 97%  
- **AUC Score:** 0.9985  
- **Calibration Error:** 0.0484  

## 📂 Repository Structure
- **`src/`** → Python scripts for model training, evaluation, and inference.
- **`notebooks/`** → Jupyter notebooks for EDA, training, and evaluation.
- **`saved_models/`** → Trained models (`.pth` format).
- **`results/`** → Accuracy reports and calibration plots.

## 🔧 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/Alzheimer_Detection_Model.git
cd Alzheimer_Detection_Model
pip install -r requirements.txt

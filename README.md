📜 README.md (Detailed Documentation)

# Aerial Semantic Segmentation using U-Net in PyTorch

## 📌 Overview
This repository provides a **U-Net** model implementation for **semantic segmentation** of aerial images using **PyTorch**. The model is trained on aerial images with multiple classes such as roads, buildings, vegetation, etc.

## 📂 Repository Structure
```bash 
📂 aerial-semantic-segmentation 
 │── 📂 data/
 │── 📂 models/ # Trained model weights
 │── 📂 notebooks/ # Jupyter notebook for visualization
 │── 📂 src/ # Source code
 │ │── dataset.py # Data loading
 │ │── model.py # U-Net model
 │ │── train.py # Training script
 │ │── predict.py # Prediction script
 │── 📂 results/ # Visualized outputs
 │── 📜 requirements.txt # Required Python libraries
 │── 📜 README.md # Documentation
 │── 📜 .gitignore # Ignore large files
```

## Table of Contents  
- [Aerial Semantic Segmentation using U-Net in PyTorch](#aerial-semantic-segmentation-using-u-net-in-pytorch)
  - [📌 Overview](#-overview)
  - [📂 Repository Structure](#-repository-structure)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [🚀 Installation](#-installation)
    - [1️⃣ Clone the Repository](#1️⃣-clone-the-repository)
    - [Install Dependencies:](#install-dependencies)
  - [Usage](#usage)
    - [Model Architecture:](#model-architecture)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Results](#results)
  - [References](#references)


## Introduction  
Semantic segmentation is a deep learning technique that assigns a class label to every pixel in an image. This project uses **U-Net**, a widely used architecture for segmentation tasks. The goal is to train a model that can identify different objects in aerial images, such as **buildings, trees, roads, and vehicles**.

## Dataset  
- [Semantic Segmentation Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset/data)

The dataset used is the **Semantic Drone Dataset**, which contains:  
- **Original aerial images** (JPG format)  
- **Semantic label images** (PNG format)  
- **Class dictionary CSV** mapping labels to RGB colors  

### Demo Images and their segmentation masks:
![Semantic masks](https://github.com/Geetha-Shashank-Pericherla/Aerial-Semantic-Segmentation/blob/main/results/example_images.png)


## 🚀 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Geetha-Shashank-Pericherla/Aerial-Semantic-Segmentation.git
cd Aerial-Semantic-Segmentation
```

### Install Dependencies:
To set up the environment, install the required dependencies:
```bash
pip install -r requirements.txt
```
or 
```bash
pip install torch torchvision segmentation-models-pytorch numpy pandas matplotlib opencv-python
```

## Usage
1. Run the Notebook
Open and execute semantic_segmentation.ipynb to train the model and evaluate its performance.

2. **Training**
To train the model, run:
```bash
python train.py
```

3. Evaluation
To test the trained model, run:
```bash
python evaluate.py
```

### Model Architecture:
The model is based on U-Net, which consists of:
- Encoder (Contracting Path): A series of convolutional layers followed by max-pooling.
- Bottleneck: The lowest level of the U-Net before upsampling.
- Decoder (Expanding Path): Upsampling layers to restore image size, with skip connections from the encoder.


### Training
The model is trained using:
- Adam optimizer
- Binary Cross-Entropy (BCE) loss with Tversky Loss
- Batch size = 4
- 15 epochs

### Evaluation
After training, the model is evaluated using:
- Pixel accuracy
- Intersection over Union (IoU) score

## Results
- The model segments aerial images into different classes.
- Sample outputs include original images, ground truth masks, and predicted masks.

### Sample Image
![Original Image](https://github.com/Geetha-Shashank-Pericherla/Aerial-Semantic-Segmentation/blob/main/results/validation_image.png)

### Output mask for each class:
![Output masks](https://github.com/Geetha-Shashank-Pericherla/Aerial-Semantic-Segmentation/blob/main/results/output_mask_for_each_class.png)


## References
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset/data)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)


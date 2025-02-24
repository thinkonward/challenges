# 3D Seismic Volume Segmentation using a UNet-Based Architecture  

---

## Table of Contents  
1. [Introduction](#introduction)  
2. [Methodology](#methodology)  
   - [Data Preprocessing](#data-preprocessing)  
   - [Model Architecture](#model-architecture)  
3. [Loss Function](#loss-function)  
4. [Training Setup](#training-setup)  
   - [Hardware and Constraints](#hardware-and-constraints)  
   - [Optimization and Precision](#optimization-and-precision)  
5. [Results and Discussion](#results-and-discussion)  
6. [Limitations and problems](#limitations-and-future-work)

---

## Introduction  

This project leverages a deep learning approach based on a UNet-inspired architecture to address this problem. The primary focus was to balance segmentation performance with computational efficiency under constrained hardware resources.

---

## Methodology  

### Data Preprocessing  
The input data of the model consisted of seismic volumes in `uint8` format to minimize memory usage. Due to hardware limitations, mixed precision training was employed, significantly reducing GPU memory requirements while maintaining computational speed.  

Key preprocessing steps included:  
- Use log1p to normalize the high range values.
- Use sign to keep the negative values.
- Use a simple linear transformation to use almost all the 0-255 range.

### Model Architecture  
The model design was based on a 2D U-Net architecture with several enhancements to improve segmentation accuracy:  
- **3D Convolutional Initial Module**: The initial layers utilized 3D convolutions to capture volumetric features effectively.  
- **Refinement Head**: A dedicated refinement head improved the model's ability to capture fine details in the segmentation masks.  
- **Internal Guidance Modules**: Auxiliary supervision was introduced via internal guidance modules, aiding intermediate layers in learning meaningful features and improving gradient flow.  

### Inference Pipeline
The inference pipeline is performed with a sliding window approach in order to avoid border artifacts.

---


## Loss Function  

To address the challenges of class imbalance and improve segmentation quality, a combination of two loss functions was employed:  
1. **Dice Loss**: To optimize overlap between predicted and ground truth masks, especially for imbalanced classes.  
2. **Binary Cross-Entropy (BCE)**: To penalize pixel-level prediction errors, providing a more granular optimization signal.

---

## Training Setup  

### Hardware and Constraints  
The model was trained on a system with the following specifications:  
- GPU: NVIDIA RTX 3080 (10GB VRAM)  
- RAM: 64GB  

### Optimization and Precision  
Given the memory constraints, the following strategies were adopted:  
- **Mixed Precision Training**: Utilizing half-precision (FP16) operations to reduce memory usage and increase training speed.  
- **Batch Size Tuning**: Carefully chosen batch sizes to avoid GPU memory overflow while ensuring stable training.  
- **unint8 Data Format**: Using `uint8` data format for input data to minimize memory usage.

---


## Limitations and problems

### Limitations  
- **Computational Constraints**: Training was limited by GPU memory and processing power, restricting batch size and model complexity.  
- **Precision Limitations**: The use of `uint8` data format and mixed precision training, while efficient, may have impacted overall model accuracy.  

## Potencial problems
- The the model was created with tensorflow 2.7.1. It is necessary that the inference environment has the correct nvidia drivers in order to install the cuda12. 
- The cuda12 is automatically installed by tensorflow[and_cuda]==2.17.1, however, it is necessary to have the correct nvidia drivers that can not be installed via pip.
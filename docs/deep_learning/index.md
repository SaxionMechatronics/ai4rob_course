# Deep Learning Module

## Course Title
**In-Depth Coding: Building a Deep Neural Network (DNN) from End to End**

## Purpose
This course provides a structured, experiment-driven journey through the **entire process of developing, training, and optimizing a deep neural network (DNN)** from scratch.  
Each module builds upon the previous one, illustrating how incremental architectural and training modifications impact performance, generalization, and computational efficiency.

The lessons are designed to be **self-contained, progressive, and practical**, making it easy to understand both the conceptual reasoning and the coding implementation behind each stage.

---

# Topics Covered

| Module | Focus Area | Key Concepts | Expected Outcome |
|---------|-------------|---------------|------------------|
| **Code 1 – Model Setup** | Initial Environment Setup | Data transforms, loaders, baseline training/testing | Establish functional training loop and baseline performance |
| **Code 2 – Model Skeleton** | Core Architecture | Simplified base network design | Create a reusable, consistent model structure |
| **Code 3 – Lighter Model** | Parameter Reduction | Model compactness, efficiency | Achieve similar accuracy with fewer parameters |
| **Code 4 – Batch Normalization** | Training Stability | BatchNorm layers, faster convergence | Improve learning efficiency and accuracy |
| **Code 5 – Regularization** | Overfitting Control | Dropout layers, regularization theory | Enhance generalization and prevent overfitting |
| **Code 6 – Global Average Pooling (GAP)** | Architectural Simplification | GAP layer, removing dense kernels | Reduce model complexity while maintaining performance |
| **Code 7 – Increasing Capacity** | Balanced Expansion | Deeper architecture, added layers | Recover performance by controlled capacity increase |
| **Code 8 – Correct MaxPooling Location** | Receptive Field Optimization | Pooling placement, Dropout tuning | Improve feature abstraction and stability |
| **Code 9 – Image Augmentation** | Data Diversity | Rotation, transformation, augmentation | Strengthen robustness and reduce underfitting |
| **Code 10 – Learning Rate Scheduling** | Convergence Optimization | Step-based LR scheduling | Achieve stable high accuracy efficiently |

---

## Core Learning Dimensions
1. **Architectural Design** – How convolutional layers, pooling, and normalization influence feature extraction and model efficiency.  
2. **Regularization Techniques** – Methods to prevent overfitting and balance capacity vs. generalization.  
3. **Training Dynamics** – The importance of learning rate control, optimizers, and schedulers.  
4. **Data Engineering** – Using augmentation to enhance diversity and improve model robustness.  
5. **Performance Analysis** – Interpreting accuracy, loss, and parameter trade-offs across model versions.  

---

## Progression Summary
The course starts with a **large, over-parameterized model (6.3M parameters)** and methodically refines it to a **compact, efficient version (~13.8k parameters)** that still achieves **~99.5% accuracy**.  
By the end, learners gain a deep understanding of the engineering decisions that drive real-world neural network optimization.

---

## End Goal
Design a compact DNN that meets the following performance criteria:

| Criterion | Target |
|-----------|---------|
| **Test Accuracy** | ≥ 99.4% (consistent across final epochs) |
| **Training Epochs** | ≤ 15 |
| **Total Parameters** | ≤ 8,000 |

---

> **This overview provides the conceptual map for the entire learning journey.**  
> Each subsequent section (Code 1–10) will dive deeper into the *how* and *why* behind every improvement.


## Module Contents

- [Introduction to Git | Python 101 | Pytorch 101](0_Git_Python_Pytorch.md) - Basic intution into Git, Python and Pytorch 
- [Introduction to Deep Learning](12_Introduction.md) - Overview of course
- [Code 1](1_Code.md) - Model Setup
- [Quiz 1](1_mcq.md)
- [Code 2](2_Code.md) - Model Skeleton
- [Quiz 2](2_mcq.md)
- [Code 3](3_Code.md) - Lighter Model
- [Quiz 3](3_mcq.md)
- [Code 4](4_Code.md) - Batch Normalization Integration
- [Quiz 4](4_mcq.md)
- [Code 5](5_Code.md) - Regularization
- [Quiz 5](5_mcq.md)
- [Code 6](6_Code.md) - Global Average Pooling
- [Quiz 6](6_mcq.md)
- [Code 7](7_Code.md) - Increasing Model Capacity
- [Quiz 7](7_mcq.md)
- [Code 8](8_Code.md) - Correcting MaxPooling Location
- [Quiz 8](8_mcq.md)
- [Code 9](9_Code.md) - Image Augmentation
- [Quiz 9](9_mcq.md)
- [Code 10](10_Code.md) - Learning Rate Scheduling
- [Quiz 10](10_mcq.md)
- [Summary](11_Summary.md) - Summary
- [Quiz](mcq.md)
- [Backward Propagation](Backward_Propagation.md) - Inroducion to Backward Propagation

 
---

[Start with Introduction to Git | Python 101 | Pytorch 101](0_Git_Python_Pytorch.md){ .md-button .md-button--primary }
# Lighter Model

## Objectives
- Reduce overall model size while maintaining accuracy  
- Simplify architecture and decrease total parameter count  
- Evaluate efficiency vs. performance trade-off  
- Retain training and evaluation pipelines unchanged  

---

# Model Configuration

### Adjustments
- Reduced number of convolutional filters per layer  
- Simplified fully connected layer structure  
- Preserved model skeleton for consistency  
- Verified end-to-end functionality post-reduction  

---

# Results

| Metric | Value |
|--------|-------|
| Total Parameters | 10,700 |
| Best Training Accuracy | 99.00% |
| Best Test Accuracy | 98.98% |

---

# Analysis

### Observations
- Significant parameter reduction (194k → 10.7k) with minimal accuracy loss  
- Model generalizes well and shows no signs of overfitting  
- Indicates an efficient baseline suitable for additional improvements  

---

# Next Steps
- Introduce normalization to accelerate convergence  
- Test model robustness under different batch sizes  
- Compare performance-per-parameter ratio with previous version  
- Begin experimenting with advanced regularization in later stages  


---
## Link to the Google Colab 

<a target="_blank" href="https://colab.research.google.com/github/SaxionMechatronics/ai4rob_course/tree/main/docs/notebooks/deep_learning/Code_3.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
---
## Check your understanding
[Quiz 3](3_mcq.md){ .md-button }
---
[← Back to Model Skeleton](2_Code.md){ .md-button }
[Start with Batch Normalization Integration →](4_Code.md){ .md-button .md-button--primary }
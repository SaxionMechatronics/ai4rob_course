# Batch Normalization Integration

## Objectives
- Introduce Batch Normalization layers for faster and more stable training  
- Improve gradient flow and enable higher learning rates  
- Observe effects on accuracy and generalization  

---

# Model Configuration

### Adjustments
- Added BatchNorm after convolutional layers  
- Kept architecture lightweight (≈ 10.9k parameters)  
- Maintained same optimizer and data pipeline for fair comparison  

---

# Results

| Metric | Value |
|--------|-------|
| Total Parameters | 10,900 |
| Best Training Accuracy | 99.90% |
| Best Test Accuracy | 99.30% |

---

# Analysis

### Observations
- Training accuracy improves markedly due to normalization  
- Minor overfitting appears as model learns faster  
- Confirms BatchNorm’s effectiveness in smaller networks  

---

# Next Steps
- Apply dropout to mitigate overfitting  
- Monitor training and validation losses per epoch  
- Combine BatchNorm with other regularization techniques  
- Document convergence rate changes for comparison  


---
## Link to the Google Colab 

<a target="_blank" href="https://colab.research.google.com/github/SaxionMechatronics/ai4rob_course/tree/main/docs/notebooks/deep_learning/Code_4.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
---
## Check your understanding
[Quiz 4](4_mcq.md){ .md-button }
---

[← Back to Lighter Model](3_Code.md){ .md-button }
[Start with BRegularization →](5_Code.md){ .md-button .md-button--primary }
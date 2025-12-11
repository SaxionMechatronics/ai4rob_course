# Image Augmentation

## Objectives
- Enhance model robustness via rotation-based augmentation  
- Compensate for limited dataset variability  
- Evaluate influence on training dynamics and test accuracy  

---

# Model Configuration

### Adjustments
- Applied rotation between 5° – 7° on training images  
- Retained identical architecture (13.8k parameters)  
- Re-trained model to convergence  

---

# Results

| Metric | Value |
|--------|-------|
| Total Parameters | 13,800 |
| Best Training Accuracy | 99.15% |
| Best Test Accuracy | 99.50% (18th Epoch) |

---

# Analysis

### Observations
- Training accuracy slightly reduced due to harder data  
- Test accuracy improved, indicating stronger generalization  
- Data augmentation effectively reduces overfitting  

---

# Next Steps
- Add additional transformations (shear, scale, brightness)  
- Compare augmentation-on vs. augmentation-off performance curves  
- Log augmented samples to validate transform integrity  

---
## Link to the Google Colab 
<a target="_blank" href="https://colab.research.google.com/github/SaxionMechatronics/ai4rob_course/blob/main/docs/notebooks/deep_learning/Code_9.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---
## Check your understanding
[Quiz 9](9_mcq.md){ .md-button }
---

[← Back to Correcting MaxPooling Location](8_Code.md){ .md-button }
[Start with Learning Rate Scheduling →](10_Code.md){ .md-button .md-button--primary }
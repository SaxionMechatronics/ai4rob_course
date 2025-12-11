# Correcting MaxPooling Location

## Objectives
- Place MaxPooling at correct receptive field (RF = 5)  
- Add new layer after GAP to expand capacity  
- Apply Dropout uniformly across layers  

---

# Model Configuration

### Adjustments
- Revised MaxPooling placement based on RF analysis  
- Ensured consistent Dropout after each convolutional block  
- Added post-GAP layer for richer features  

---

# Results

| Metric | Value |
|--------|-------|
| Total Parameters | 13,800 |
| Best Training Accuracy | 99.39% |
| Best Test Accuracy | 99.41% (9th Epoch) |

---

# Analysis

### Observations
- Model generalizes well with balanced train/test performance  
- Overfitting minimized through uniform regularization  
- Achieves consistent 99.4 % accuracy range  

---

# Next Steps
- Introduce light image augmentation for further improvement  
- Test pooling alternatives (average vs max)  
- Examine learning-curve stability across multiple runs  


---
## Link to the Google Colab 
<a target="_blank" href="https://colab.research.google.com/github/SaxionMechatronics/ai4rob_course/blob/main/docs/notebooks/deep_learning/Code_8.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
---
## Check your understanding
[Quiz 8](8_mcq.md){ .md-button }
---
[← Back to Increasing Model Capacity](7_Code.md){ .md-button }
[Start with Image Augmentation →](9_Code.md){ .md-button .md-button--primary }
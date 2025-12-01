# Increasing Model Capacity

## Objectives
- Compensate for performance loss after GAP by adding capacity  
- Add additional layers after GAP to improve representation power  
- Maintain efficient parameter usage  

---

# Model Configuration

### Adjustments
- Introduced new convolutional layer(s) post-GAP  
- Expanded channel depth in later stages  
- Retained BatchNorm + Dropout combination  

---

# Results

| Metric | Value |
|--------|-------|
| Total Parameters | 11,900 |
| Best Training Accuracy | 99.33% |
| Best Test Accuracy | 99.04% |

---

# Analysis

### Observations
- Model shows slight overfitting; Dropout placement may need adjustment  
- Added layers successfully recover lost accuracy  
- Indicates end-layer capacity is influential in MNIST-scale tasks  

---

# Next Steps
- Re-evaluate Dropout locations and probabilities  
- Analyze receptive field coverage (≈ 5×5 RF patterns visible)  
- Test adding one more layer after GAP to further enhance capacity  

---
## Link to the Google Colab 

<a target="_blank" href="https://colab.research.google.com/github/SaxionMechatronics/ai4rob_course/tree/main/docs/notebooks/deep_learning/Code_7.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
---

## Check your understanding
[Quiz 7](7_mcq.md){ .md-button }
---


[← Back to Global Average Pooling](6_Code.md){ .md-button }
[Start with Correcting MaxPooling Location →](8_Code.md){ .md-button .md-button--primary }
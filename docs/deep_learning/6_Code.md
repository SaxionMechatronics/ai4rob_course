# Global Average Pooling

## Objectives
- Replace large final convolutional kernel with Global Average Pooling (GAP)  
- Reduce parameter count further while simplifying architecture  
- Evaluate performance differences between heavy and light endings  

---

# Model Configuration

### Adjustments
- Implemented GAP layer before final classifier  
- Removed dense layer with large kernel size  
- Achieved significant reduction in total parameters  

---

# Results

| Metric | Value |
|--------|-------|
| Total Parameters | 6,000 |
| Best Training Accuracy | 99.86% |
| Best Test Accuracy | 98.13% |

---

# Analysis

### Observations
- Accuracy drop expected due to lower capacity  
- GAP improves architectural efficiency and reduces computation cost  
- Confirms capacity–performance trade-off relationship  

---

# Next Steps
- Increase network depth slightly to regain lost accuracy  
- Consider adding a 1×1 convolution before GAP  
- Begin experimenting with feature-map visualizations for interpretability  

---
## Link to the Google Colab 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaxionMechatronics/ai4rob_course/blob/main/docs/notebooks/deep_learning/code6.ipynb)
---
## Check your understanding
[Quiz 6](6_mcq.md){ .md-button }
---


[← Back to Regularization](5_Code.md){ .md-button }
[Start with Increasing Model Capacity →](7_Code.md){ .md-button .md-button--primary }
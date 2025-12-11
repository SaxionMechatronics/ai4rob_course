# Regularization

## Objectives
- Add Dropout to reduce overfitting and enhance generalization  
- Retain same parameter count to isolate Dropout’s effect  
- Assess impact on convergence and accuracy  

---

# Model Configuration

### Adjustments
- Introduced Dropout layers after convolutional blocks  
- Maintained total parameters (~ 10.9k)  
- Trained for 25 epochs to evaluate stability  

---

# Results

| Metric | Value |
|--------|-------|
| Total Parameters | 10,900 |
| Best Training Accuracy | 99.47% |
| Best Test Accuracy | 99.30% |

---

# Analysis

### Observations
- Regularization works effectively; test accuracy stabilized  
- Training accuracy slightly reduced — expected for Dropout usage  
- Model capacity still limits further gains; architecture refinement needed  

---

# Next Steps
- Replace large kernel with Global Average Pooling to simplify design  
- Tune dropout rate (0.1 – 0.3) and record accuracy trends  
- Explore additional regularization such as weight decay  

---
## Link to the Google Colab 
<a target="_blank" href="https://colab.research.google.com/github/SaxionMechatronics/ai4rob_course/blob/main/docs/notebooks/deep_learning/Code_5.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---
## Check your understanding
[Quiz 5](5_mcq.md){ .md-button }
---

[← Back to Batch Normalization Integration](4_Code.md){ .md-button }
[Start with Global Average Pooling →](6_Code.md){ .md-button .md-button--primary }
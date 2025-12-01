# Learning Rate Scheduling

## Objectives
- Introduce a Learning Rate (LR) Scheduler to optimize convergence  
- Reduce learning rate dynamically after defined epochs  
- Observe improvements in stability and final accuracy  

---

# Model Configuration

### Adjustments
- Implemented step-based LR reduction (÷ 10 after 6th epoch)  
- Maintained 13.8k-parameter architecture  
- Trained for 20 epochs to monitor LR effects  

---

# Results

| Metric | Value |
|--------|-------|
| Total Parameters | 13,800 |
| Best Training Accuracy | 99.21% |
| Best Test Accuracy | 99.45% (9th Epoch), 99.48% (20th Epoch) |

---

# Analysis

### Observations
- Scheduler accelerates early convergence  
- Plateau accuracy similar to previous best (~ 99.5%)  
- Demonstrates importance of fine-tuned LR strategy  

---
## Link to the Google Colab 

<a target="_blank" href="https://colab.research.google.com/github/SaxionMechatronics/ai4rob_course/tree/main/docs/notebooks/deep_learning/Code_10.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
---

[← Back to Image Augmentation](9_Code.md){ .md-button }
[Start with Summary →](11_Summary.md){ .md-button .md-button--primary }

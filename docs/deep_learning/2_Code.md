# Model Skeleton

## Objectives
- Establish a clean and reusable model skeleton  
- Maintain structural consistency across future experiments  
- Keep the implementation minimal without adding complexity  
- Serve as the foundation for subsequent architectural improvements  

---

# Model Configuration

### Skeleton Implemented
- Core model structure finalized (input → convolutional blocks → classifier)  
- Avoided unnecessary layers or parameters  
- Forward and backward passes verified  
- Training and evaluation pipelines remain unchanged  
- Acts as a stable baseline for further refinements  

---

# Results

| Metric | Value |
|--------|-------|
| Total Parameters | 194,000 |
| Best Training Accuracy | 99.35% |
| Best Test Accuracy | 99.02% |

---

# Analysis

### Observations
- Parameter count reduced significantly from the initial setup (6.3M → 194k)  
- Model achieves strong accuracy while being much smaller  
- Slight overfitting persists, suggesting potential need for regularization  
- Architecture demonstrates good balance between performance and simplicity  

---

# Next Steps
- Investigate which layers contribute most to overfitting  
- Introduce dropout or weight decay to improve generalization  
- Visualize feature maps to confirm proper layer activations  
- Prepare for parameter reduction and normalization layers in the next stage  
- Use this skeleton as a consistent base for all future experiments  

---

## Link to the Google Colab 

<a target="_blank" href="https://colab.research.google.com/github/SaxionMechatronics/ai4rob_course/blob/main/docs/notebooks/deep_learning/Code_2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
---
---
## Check your understanding
[Quiz 2](2_mcq.md){ .md-button }

[← Back to Model Skeleton](1_Code.md){ .md-button }
[Start with Lighter Model →](3_Code.md){ .md-button .md-button--primary }
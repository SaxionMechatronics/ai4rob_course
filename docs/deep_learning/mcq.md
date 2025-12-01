## Question 1
```mcq
---
type: single
question: What is the role of a baseline model in a deep learning project?
---
- [ ] Final optimized model  
- [x] A reference architecture to compare later improvements  
  > The baseline acts as a fixed reference to evaluate future architectural and training changes.  
- [ ] Model used only for validation  
- [ ] Pretrained model on ImageNet  
 
```

---

## Question 2
```mcq
---
type: single
question: Why do large models with millions of parameters often overfit small datasets?
---
- [x] They memorize training data due to excessive capacity  
  > When a model has too many parameters relative to dataset size, it learns noise rather than general patterns.  
- [ ] They have too few features  
- [ ] They use small kernels  
- [ ] They skip normalization  


```

---

## Question 3
```mcq
---
type: single
question: What is the purpose of establishing a model “skeleton” before experimentation?
---
- [x] To ensure consistent structure for controlled experiments  
  > A skeleton provides a reproducible base so that later architectural changes can be objectively compared.  
- [ ] To maximize training speed  
- [ ] To randomize layer order  
- [ ] To perform data augmentation  


```

---

## Question 4
```mcq
---
type: single
question: Why is reducing model parameters often beneficial?
---
- [x] It helps reduce overfitting and improves generalization  
  > Smaller models are less likely to memorize data and usually generalize better on unseen inputs.  
- [ ] It increases computational cost  
- [ ] It always increases accuracy  
- [ ] It improves visualization quality  

```

---

## Question 5
```mcq
---
type: single
question: What is the main purpose of Batch Normalization in neural networks?
---
- [x] To stabilize learning by normalizing activations  
  > Batch Normalization reduces internal covariate shift, leading to faster and more stable convergence.  
- [ ] To increase regularization strength  
- [ ] To freeze gradients  
- [ ] To modify optimizer behavior  

```

## Question 6
```mcq
---
type: single
question: Why can Batch Normalization sometimes increase overfitting?
---
- [x] It allows faster convergence, causing the model to memorize data more easily  
  > Rapid learning due to stabilized gradients may amplify overfitting if no regularization is applied.  
- [ ] It randomly drops neurons  
- [ ] It reduces model size too much  
- [ ] It introduces noisy gradients   
 
```

---

## Question 7
```mcq
---
type: single
question: How does Dropout help improve model generalization?
---
- [x] By randomly deactivating neurons during training  
  > Dropout prevents co-adaptation of neurons, forcing the model to learn redundant, robust features.  
- [ ] By increasing learning rate dynamically  
- [ ] By freezing selected layers  
- [ ] By normalizing weights  

```

---

## Question 8
```mcq
---
type: single
question: What is the main effect of Global Average Pooling in a CNN?
---
- [x] It replaces dense layers to reduce parameters while preserving spatial information  
  > GAP averages feature maps, lowering parameter count and reducing overfitting risk.  
- [ ] It adds additional convolutional filters  
- [ ] It increases receptive field size  
- [ ] It removes spatial structure completely  

```

---

## Question 9
```mcq
---
type: single
question: Why is proper MaxPooling placement important in CNN architectures?
---
- [x] It ensures correct receptive field coverage and hierarchical feature extraction  
  > Proper pooling order maintains spatial hierarchy and prevents feature loss.  
- [ ] It increases GPU memory usage  
- [ ] It improves training speed only  
- [ ] It has no real impact on performance  

```

---

## Question 10
```mcq
---
type: single
question: How does data augmentation improve model performance?
---
- [x] It introduces variations that help the model generalize better to unseen data  
  > Augmentations like rotation, shift, or flip prevent the network from overfitting to training samples.  
- [ ] It increases dataset redundancy  
- [ ] It reduces training speed  
- [ ] It adds noise to labels  
```

## Question 11
```mcq
---
type: single
question: Why might increasing model capacity improve accuracy in certain cases?
---
- [x] It allows the network to capture more complex representations of data  
  > When a model is too simple, it may underfit; more capacity helps learn richer feature hierarchies.  
- [ ] It prevents overfitting automatically  
- [ ] It reduces the need for regularization  
- [ ] It slows down gradient updates intentionally  
 
```

---

## Question 12
```mcq
---
type: single
question: Which part of a CNN architecture typically has the greatest influence on classification accuracy?
---
- [x] The final fully connected or classification layers  
  > The end layers aggregate learned features and directly impact prediction performance.  
- [ ] The input normalization layer  
- [ ] The pooling layers  
- [ ] The data loader configuration  

```

---

## Question 13
```mcq
---
type: single
question: What is the main reason to use a Learning Rate Scheduler during training?
---
- [x] To reduce the learning rate over time for stable convergence  
  > Gradually lowering the LR helps fine-tune near the loss minimum, improving stability and accuracy.  
- [ ] To randomly change the learning rate each epoch  
- [ ] To speed up training by increasing LR  
- [ ] To disable gradient updates periodically  

```

---

## Question 14
```mcq
---
type: single
question: What problem can occur if the learning rate is too high?
---
- [x] The model may diverge or oscillate around the minimum  
  > A large LR overshoots the optimal point, preventing proper convergence.  
- [ ] The model underfits slowly  
- [ ] Training stops immediately  
- [ ] Gradients vanish completely  

```

---

## Question 15
```mcq
---
type: single
question: How does reducing the learning rate later in training improve results?
---
- [x] It allows finer weight adjustments for improved generalization  
  > Lowering LR after initial convergence helps the model settle into a stable minimum.  
- [ ] It increases gradient magnitude  
- [ ] It resets optimizer states  
- [ ] It prevents backpropagation entirely  

```

## Question 16
```mcq
---
type: single
question: Why does Dropout sometimes reduce training accuracy but improve test accuracy?
---
- [x] It prevents overfitting by randomly disabling parts of the network  
  > Dropout forces general feature learning instead of memorization, leading to better generalization.  
- [ ] It increases regularization loss  
- [ ] It slows batch normalization  
- [ ] It introduces bias in weight updates  

```

---

## Question 17
```mcq
---
type: single
question: How does Global Average Pooling differ from Flatten + Dense layers?
---
- [x] GAP uses averaging instead of trainable connections, reducing parameters significantly  
  > GAP minimizes the number of weights and lowers overfitting risk compared to dense layers.  
- [ ] GAP adds additional bias terms  
- [ ] GAP multiplies feature maps by filters  
- [ ] GAP removes activation functions  

```

---

## Question 18
```mcq
---
type: single
question: What happens if MaxPooling is applied too early in a CNN?
---
- [x] Important spatial features may be lost prematurely  
  > Early pooling discards fine details that are crucial for learning complex patterns.  
- [ ] It increases computation time  
- [ ] It enhances feature granularity  
- [ ] It doubles the receptive field automatically  

```

---

## Question 19
```mcq
---
type: single
question: Why does data augmentation improve robustness to unseen images?
---
- [x] It exposes the model to diverse transformations of the same data  
  > Training on rotated, flipped, or shifted images teaches invariance to real-world variations.  
- [ ] It decreases effective dataset size  
- [ ] It alters the optimizer dynamics  
- [ ] It freezes convolutional kernels  

```

---

## Question 20
```mcq
---
type: single
question: What overall benefit does combining normalization, dropout, and LR scheduling achieve?
---
- [x] It balances convergence speed, stability, and generalization  
  > Together, these techniques control gradient flow, prevent overfitting, and fine-tune learning efficiency.  
- [ ] It reduces model depth  
- [ ] It eliminates the need for backpropagation  
- [ ] It guarantees 100% accuracy  
 
```
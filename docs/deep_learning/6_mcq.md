## Question 1
```mcq
---
type: single
question: What major architectural change is introduced in Code 6?
---
- [ ] Added more dense layers  
- [x] Replaced the fully connected layer with Global Average Pooling (GAP)  
- [ ] Increased kernel size  
- [ ] Introduced a new activation function  
  
 
```

---

## Question 2
```mcq
---
type: single
question: What is the main benefit of using Global Average Pooling (GAP)?
---
- [ ] Increases the number of parameters  
- [x] Reduces parameters and overfitting while keeping key spatial information  
- [ ] Adds regularization noise  
- [ ] Improves non-linearity  
 

```

---

## Question 3
```mcq
---
type: single
question: Why does test accuracy slightly drop after introducing GAP?
---
- [ ] Overfitting increases  
- [x] The model’s overall capacity is reduced  
- [ ] Learning rate is too high  
- [ ] Batch size is too small  

```

---

## Question 4
```mcq
---
type: single
question: Approximately how many parameters remain after applying GAP?
---
- [x] 6 k  
- [ ] 10.9 k  
- [ ] 11.9 k  
- [ ] 194 k  

```

---

## Question 5
```mcq
---
type: single
question: What is a key difference between GAP and a dense layer?
---
- [x] GAP averages spatial features and eliminates trainable weights  
- [ ] GAP multiplies all features by a learnable parameter  
- [ ] GAP adds extra normalization  
- [ ] GAP increases spatial resolution  


```
## Question 1
```mcq
---
type: single
question: What is the main difference between value-based and policy-based methods?
---
- [ ] Value-based methods learn faster, policy-based methods learn slower  
- [x] Value-based methods learn value functions then derive a policy, while policy-based methods directly learn the policy  
- [ ] Value-based methods work for continuous actions, policy-based methods work for discrete actions  
- [ ] There is no difference; they are the same approach  
```

---

## Question 2
```mcq
---
type: single
question: Which type of method is better suited for continuous action spaces like robot joint torques?
---
- [ ] Value-based methods, because they can learn Q-values for any action  
- [x] Policy-based methods, because they can directly output continuous actions without needing to maximize over infinite possibilities  
- [ ] Both work equally well for continuous actions  
- [ ] Neither works for continuous actions; you must discretize first  
```

---

## Question 3
```mcq
---
type: single
question: What is a key advantage of value-based methods over policy-based methods?
---
- [ ] They work better for continuous actions  
- [x] They are more sample efficient and can learn off-policy from any experience  
- [ ] They always converge faster  
- [ ] They require less computation  
```

---

## Question 4
```mcq
---
type: single
question: How do policy-based methods handle exploration compared to value-based methods?
---
- [ ] Policy-based methods require explicit exploration strategies like epsilon-greedy  
- [x] Policy-based methods have exploration built into the stochastic policy, while value-based methods need explicit exploration strategies  
- [ ] Both require the same exploration mechanisms  
- [ ] Policy-based methods cannot explore at all  
```

---

## Question 5
```mcq
---
type: single
question: What is the main idea behind actor-critic methods?
---
- [ ] They use only value functions without any policy  
- [ ] They use only policies without any value functions  
- [x] They combine both approaches: an actor (policy) decides actions while a critic (value function) evaluates them  
- [ ] They are a completely different approach unrelated to value or policy methods  
```


## Question 1
```mcq
---
type: single
question: What is the core idea behind REINFORCE and policy gradient methods?
---
- [ ] Learn Q-values for each state-action pair, then derive a policy
- [x] Directly learn and optimize the policy parameters to maximize expected return
- [ ] Learn a value function to estimate state values
- [ ] Use dynamic programming to compute optimal policies
```

---

## Question 2
```mcq
---
type: single
question: Why does REINFORCE use the log-probability gradient ∇_θ log π_θ(a|s) instead of the probability gradient ∇_θ π_θ(a|s)?
---
- [ ] Log-probability gradients are always larger, making learning faster
- [x] The policy gradient theorem requires this form, and it's easier to compute for neural networks
- [ ] Log-probability prevents the policy from becoming deterministic
- [ ] It's only used for discrete action spaces, not continuous ones
```

---

## Question 3
```mcq
---
type: single
question: What is the main purpose of using a baseline (like V(s_t)) in REINFORCE?
---
- [ ] To make the algorithm converge faster
- [x] To reduce the variance of gradient estimates without introducing bias
- [ ] To ensure the policy always improves
- [ ] To make the algorithm work with continuous actions
```

---

## Question 4
```mcq
---
type: single
question: What is a key advantage of policy gradient methods like REINFORCE over value-based methods like Q-Learning?
---
- [ ] They are always more sample efficient
- [x] They naturally handle continuous action spaces without needing to maximize over infinite actions
- [ ] They always converge faster
- [ ] They don't require any exploration strategy
```


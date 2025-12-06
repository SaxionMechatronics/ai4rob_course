## Question 1
```mcq
---
type: single
question: What is the Q-Learning update rule?
---
- [ ] Q(s, a) ← Q(s, a) + α[r + γ Q(s', a') - Q(s, a)] where a' is the action actually taken
- [x] Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)] where we use the maximum Q-value in the next state
- [ ] Q(s, a) ← Q(s, a) + α[r - Q(s, a)] without considering future rewards
- [ ] Q(s, a) ← r + γ max_a' Q(s', a') replacing the old value completely
```

---

## Question 2
```mcq
---
type: single
question: Why is Q-Learning considered an off-policy algorithm?
---
- [ ] Because it requires knowing the transition probabilities P(s'|s, a)
- [x] Because it learns the optimal policy while following an exploratory policy (like ε-greedy)
- [ ] Because it only works with deterministic policies
- [ ] Because it cannot learn from past experiences
```

---

## Question 3
```mcq
---
type: single
question: What is the purpose of the ε-greedy exploration strategy in Q-Learning?
---
- [ ] To always choose the best action to maximize immediate reward
- [x] To balance exploration (trying new actions) and exploitation (using learned Q-values)
- [ ] To ensure the algorithm converges faster
- [ ] To reduce the variance in Q-value estimates
```

---

## Question 4
```mcq
---
type: single
question: What is a key limitation of Q-Learning that makes it unsuitable for continuous action spaces?
---
- [ ] It requires too much memory to store Q-values
- [x] It needs to compute argmax over all actions, which is impossible or very difficult with infinite continuous actions
- [ ] It converges too slowly for continuous actions
- [ ] It cannot handle stochastic environments
```


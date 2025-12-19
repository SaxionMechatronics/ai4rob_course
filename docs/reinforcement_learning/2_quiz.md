## Question 1
```mcq
---
type: single
question: What are the five components of an MDP?
---
- [ ] States, Actions, Policy, Rewards, Discount  
- [x] States, Actions, Transitions, Rewards, Discount  
- [ ] States, Actions, Observations, Rewards, Discount  
- [ ] States, Actions, Value function, Rewards, Discount  
```

---

## Question 2
```mcq
---
type: single
question: What does the Markov property state?
---
- [ ] The next state depends on the entire history of states and actions  
- [x] The next state depends only on the current state and action, not on history  
- [ ] The next state is always deterministic  
- [ ] The next state depends only on the current action  
```

---

## Question 3
```mcq
---
type: single
question: What is the key difference between the state-value function V and the action-value function Q?
---
- [ ] V is the expected return when following a deterministic policy, while Q is the expected return when following a stochastic policy that samples actions probabilistically  
- [x] V is the expected return from a state following a policy, while Q is the expected return from a state after taking a specific action then following the policy  
- [ ] V is the expected return computed over continuous state spaces, while Q is the expected return computed over discrete state spaces and finite action sets  
- [ ] There is no meaningful difference between them; they both represent the expected return from a state and can be used interchangeably in all RL algorithms  
```

---

## Question 4
```mcq
---
type: single
question: Why is a discount factor less than 1 necessary for infinite-horizon problems?
---
- [ ] It makes the agent prefer immediate rewards over future rewards  
- [ ] It allows the agent to forget past experiences  
- [x] It ensures the infinite sum of discounted rewards converges mathematically  
- [ ] It prevents the agent from exploring too much  
```

---

## Question 5
```mcq
---
type: single
question: What is the main difference between a state and an observation in robotics?
---
- [ ] States are discrete variables that can only take on finite values, while observations are continuous variables that the robot's sensors measure in real-time  
- [x] States contain complete information about the environment, while observations are partial and noisy measurements that the agent actually perceives  
- [ ] States are the variables that the agent can directly control through its actions, while observations are the variables that the environment determines independently  
- [ ] There is no meaningful difference between states and observations; they both refer to the same information and can be used interchangeably in all contexts  
```


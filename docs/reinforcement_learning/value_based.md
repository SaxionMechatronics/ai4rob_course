# Q-Learning: Value-Based Reinforcement Learning

## Introduction

Q-Learning is one of the most important algorithms in reinforcement learning. It's elegant, powerful, and surprisingly simple!

**Core idea:** Learn the value of taking each action in each state, then act greedily by choosing the best action.

## The Q-Function: What Are We Learning?

Recall the **action-value function**:

\[
Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid S_0 = s, A_0 = a\right]
\]

**Interpretation:** "If I'm in state \( s \), take action \( a \), then follow policy \( \pi \), what total discounted reward will I get?"

**The optimal Q-function** \( Q^*(s, a) \) tells us the best possible value:

\[
Q^*(s, a) = \max_\pi Q^\pi(s, a)
\]

**Key insight:** If we know \( Q^*(s, a) \), the optimal policy is trivial:

\[
\pi^*(s) = \arg\max_a Q^*(s, a)
\]

Just pick the action with the highest Q-value!

## The Bellman Optimality Equation for Q

The optimal Q-function satisfies the **Bellman optimality equation**:

\[
Q^*(s, a) = \mathbb{E}_{s' \sim P(s'|s,a)}\left[R(s, a, s') + \gamma \max_{a'} Q^*(s', a')\right]
\]

**Intuition:** The value of taking action \( a \) in state \( s \) equals:
1. The immediate reward \( R(s, a, s') \)
2. Plus the discounted value of the best action in the next state

This gives us a way to iteratively improve our estimates!

## Q-Learning Algorithm

### The Update Rule

Q-Learning uses the Bellman equation to update Q-values:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]
\]

Where:
- \( \alpha \) is the **learning rate** (step size)
- \( r \) is the **immediate reward**
- \( \gamma \) is the **discount factor**
- The term in brackets is called the **temporal difference (TD) error**

### Understanding the Update

Let's break down the TD error:

\[
\delta = \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{TD target}} - \underbrace{Q(s, a)}_{\text{Current estimate}}
\]

- **TD target**: Our new estimate of \( Q(s, a) \) based on the reward we just received and the best action in the next state
- **Current estimate**: What we currently think \( Q(s, a) \) is
- **TD error**: How wrong our current estimate was

We move our estimate in the direction of the TD target:

```
New estimate = Old estimate + learning_rate × TD_error
```

### The Complete Algorithm

```python
# Pseudocode for Q-Learning

Initialize Q(s, a) arbitrarily for all s, a
Set Q(terminal_state, *) = 0

For each episode:
    Initialize state s
    
    While s is not terminal:
        # Choose action (ε-greedy)
        With probability ε:
            a = random action
        Otherwise:
            a = argmax_a' Q(s, a')
        
        # Take action, observe outcome
        Take action a, observe reward r and next state s'
        
        # Q-Learning update
        Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
        
        # Move to next state
        s ← s'
```

### Key Properties

✅ **Off-policy**: Learns optimal policy while following exploratory policy (ε-greedy)

✅ **Model-free**: Doesn't need to know transition probabilities \( P(s'|s, a) \)

✅ **Converges to optimal**: Under certain conditions (visiting all state-action pairs infinitely often, appropriate learning rate schedule)

## Exploration: The ε-Greedy Policy

Q-Learning needs to **explore** to discover good actions, but also **exploit** what it has learned.

**ε-greedy policy:**

\[
\pi(s) = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon
\end{cases}
\]

**Typical schedule:**
- Start with high ε (e.g., 1.0): Explore extensively at first
- Decay ε over time (e.g., to 0.01): Exploit more as we learn
- Keep small ε forever: Continue exploring slightly to adapt to changes

```python
# Epsilon decay schedule
epsilon = max(epsilon_min, epsilon_start * epsilon_decay ** episode)
```

## Worked Example: Grid World

Let's walk through Q-Learning on a simple 3×3 grid world.

**Setup:**
- Start: Bottom-left (0, 0)
- Goal: Top-right (2, 2), reward = +10
- Each step: reward = -1 (encourages finding shortest path)
- Actions: {up, down, left, right}
- Discount factor: γ = 0.9
- Learning rate: α = 0.1

**Initial Q-table:** All zeros

```
State | Up   | Down | Left | Right
------|------|------|------|------
(0,0) | 0.0  | 0.0  | 0.0  | 0.0
(0,1) | 0.0  | 0.0  | 0.0  | 0.0
...
```

### Episode 1: Random Exploration

**Step 1:** State (0,0), choose random action: Right
- Take action, observe: s' = (1,0), r = -1
- Update:
  \[
  Q((0,0), \text{right}) = 0 + 0.1[-1 + 0.9 \times 0 - 0] = -0.1
  \]

**Step 2:** State (1,0), choose random action: Up
- Take action, observe: s' = (1,1), r = -1
- Update:
  \[
  Q((1,0), \text{up}) = 0 + 0.1[-1 + 0.9 \times 0 - 0] = -0.1
  \]

**Continue until reaching goal...**

### After Many Episodes

The Q-table converges to show the value of each action:

```
State (1,1) - one step from goal:
  Up    = 7.9   (one step → goal, r=-1, then +10)
  Right = 7.9   (one step → goal, r=-1, then +10)
  Down  = -1.0  (moves away)
  Left  = -1.0  (moves away)

Optimal policy at (1,1): Up or Right (both lead to goal in 1 step)
```

The agent has learned that:
- Actions toward the goal have high Q-values
- Actions away from the goal have low Q-values
- The optimal path is visible from the Q-values

## Tabular vs Function Approximation

### Tabular Q-Learning

Store Q-values in a table: one entry for each (state, action) pair.

**Pros:**
- Simple to implement
- Guaranteed convergence (under conditions)
- Easy to inspect and debug

**Cons:**
- Only works for discrete, small state/action spaces
- Doesn't generalize: must visit every (s, a) pair
- Memory explodes with large spaces

**When to use:** Simple grid worlds, small discrete problems

### Function Approximation (Deep Q-Networks)

Approximate Q-function with a parameterized function (e.g., neural network):

\[
Q(s, a) \approx Q_\theta(s, a)
\]

Where \( \theta \) are the parameters (weights) of the neural network.

**Update rule:**

\[
\theta \leftarrow \theta + \alpha \left[r + \gamma \max_{a'} Q_\theta(s', a') - Q_\theta(s, a)\right] \nabla_\theta Q_\theta(s, a)
\]

**Pros:**
- Handles large/continuous state spaces
- Generalizes to unseen states
- Can learn from images, sensors, etc.

**Cons:**
- Can be unstable and diverge
- Requires careful tuning
- Loss of convergence guarantees

**When to use:** Complex state spaces (images, continuous sensors)

## Deep Q-Networks (DQN): A Brief Overview

DQN (DeepMind, 2015) made Q-Learning work with neural networks by introducing two key innovations:

### 1. Experience Replay

Store experiences \( (s, a, r, s') \) in a **replay buffer**.

Sample random mini-batches from the buffer for training.

**Why it helps:**
- Breaks correlation between consecutive samples
- Reuses data multiple times (sample efficient)
- Stabilizes training

```python
# Experience replay
replay_buffer = []

# During interaction
experience = (s, a, r, s')
replay_buffer.append(experience)

# During training
batch = random_sample(replay_buffer, batch_size)
for (s, a, r, s') in batch:
    target = r + γ * max_a' Q(s', a')
    loss = (Q(s, a) - target)²
    Update Q-network using this loss
```

### 2. Target Network

Maintain two networks:
- **Online network** \( Q_\theta \): Updated every step
- **Target network** \( Q_{\theta^-} \): Updated periodically (e.g., every 1000 steps)

Use target network for computing TD target:

\[
\text{TD target} = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
\]

**Why it helps:**
- Prevents the target from moving too quickly
- Reduces harmful correlations
- Stabilizes training

```python
# Target network update (every C steps)
if step % C == 0:
    target_network.weights = online_network.weights
```

## Simple Python Implementation

Here's a minimal Q-Learning implementation for a discrete environment:

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.Q = np.zeros((n_states, n_actions))
    
    def select_action(self, state):
        """ε-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return np.argmax(self.Q[state])  # Exploit
    
    def update(self, state, action, reward, next_state, done):
        """Q-Learning update"""
        # Current Q-value
        current_q = self.Q[state, action]
        
        # TD target
        if done:
            target_q = reward  # No future value at terminal state
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state])
        
        # Q-Learning update
        td_error = target_q - current_q
        self.Q[state, action] += self.lr * td_error
        
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, 
                          self.epsilon * self.epsilon_decay)

# Training loop
agent = QLearningAgent(n_states=100, n_actions=4)

for episode in range(1000):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Select and take action
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Update Q-values
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    # Decay exploration
    agent.decay_epsilon()
    
    print(f"Episode {episode}: Total Reward = {total_reward}, ε = {agent.epsilon:.3f}")
```

## Hyperparameters and Their Effects

### Learning Rate (α)

Controls how much we update Q-values each step.

\[
Q(s,a) \leftarrow Q(s,a) + \alpha [TD\_target - Q(s,a)]
\]

- **Too high** (e.g., α = 1.0): Forgets old information, unstable
- **Too low** (e.g., α = 0.001): Learns very slowly
- **Typical values**: 0.001 to 0.1

**Schedule:** Often decayed over time (fast learning early, refinement later)

### Discount Factor (γ)

Controls how much we value future rewards.

\[
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]
\]

- **γ = 0**: Only care about immediate reward (myopic)
- **γ = 1**: Care equally about all future rewards (far-sighted)
- **Typical values**: 0.95 to 0.99

**Effect:** Higher γ considers longer-term consequences.

### Exploration Rate (ε)

Controls exploration vs exploitation trade-off.

- **High ε** (e.g., 1.0): Random actions, lots of exploration
- **Low ε** (e.g., 0.01): Mostly greedy, little exploration
- **Typical schedule**: Start at 1.0, decay to 0.01

**Decay schedule:**
```python
epsilon = max(epsilon_min, epsilon * decay_rate)  # Exponential decay
epsilon = max(epsilon_min, epsilon - decay_step)  # Linear decay
```

## Convergence and Guarantees

Q-Learning is **proven to converge** to the optimal Q-function under these conditions:

1. **All state-action pairs visited infinitely often**
   - Ensured by sufficient exploration (ε-greedy with ε > 0)

2. **Learning rate schedule**
   - Must satisfy: \( \sum_{t=1}^{\infty} \alpha_t = \infty \) and \( \sum_{t=1}^{\infty} \alpha_t^2 < \infty \)
   - Example: \( \alpha_t = \frac{1}{t} \)

3. **Tabular representation**
   - Each (s, a) has independent Q-value
   - With function approximation, no guarantees!

**Practical note:** In practice, we use constant learning rates and function approximation. While theoretical guarantees don't hold, Q-Learning often works well!

## Q-Learning Variants

### SARSA (On-Policy Q-Learning)

Instead of using \( \max_{a'} Q(s', a') \), use the action actually taken:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]
\]

Where \( a' \) is the action selected by the current policy.

**Difference:**
- Q-Learning: Off-policy (learns optimal policy while acting exploratively)
- SARSA: On-policy (learns policy being followed, including exploration)

**When to use SARSA:** When you want the learned policy to account for exploration risk (e.g., near cliffs).

### Double Q-Learning

Q-Learning can **overestimate** values due to maximization bias.

**Solution:** Use two Q-functions, select action with one, evaluate with the other:

\[
Q_1(s, a) \leftarrow Q_1(s, a) + \alpha [r + \gamma Q_2(s', \arg\max_{a'} Q_1(s', a')) - Q_1(s, a)]
\]

**Why it helps:** Reduces overestimation, more stable learning.

### Expected SARSA

Average over all possible next actions weighted by policy:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \sum_{a'} \pi(a'|s') Q(s', a') - Q(s, a)]
\]

**Benefits:** Lower variance than SARSA, still considers exploration policy.

## Limitations of Q-Learning

❌ **Limited to discrete actions**
- Need to compute \( \arg\max_a Q(s, a) \)
- Difficult or impossible with continuous actions

❌ **Overestimation bias**
- Always taking max can lead to overly optimistic estimates
- Mitigated by Double Q-Learning

❌ **Sample inefficiency**
- Each update only affects one (s, a) pair (in tabular case)
- Needs many samples to converge

❌ **Instability with function approximation**
- Neural network Q-functions can diverge
- Requires experience replay, target networks, careful tuning

❌ **No explicit exploration strategy**
- Must add ε-greedy or other heuristics
- Not principled like entropy regularization in policy methods

## When to Use Q-Learning

### Perfect for:

✅ Discrete action spaces (games, navigation with discrete moves)

✅ Offline learning from logged data

✅ Simple environments where tabular methods suffice

✅ When you want off-policy learning

### Avoid for:

❌ Continuous action spaces (use policy gradient methods or actor-critic)

❌ Extremely large state spaces without good function approximation

❌ Real-time learning with strict sample budgets (consider policy gradients)

## Summary

**Q-Learning in a nutshell:**

1. **Learn Q-values**: Estimate the value of each action in each state
2. **Bellman updates**: Use immediate reward + value of best next action
3. **Act greedily**: Choose action with highest Q-value
4. **Explore**: Use ε-greedy to discover better strategies

**Key equation:**
\[
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
\]

**Strengths:** Simple, off-policy, proven convergence (tabular case)

**Weaknesses:** Discrete actions only, can be sample inefficient, requires careful exploration

**Modern extensions:** DQN, Double DQN, Dueling DQN, Rainbow DQN

Now let's move to the other side: **policy gradient methods** that directly learn the policy!

---

[← Back to Policy vs Value-Based](policy_vs_value.md){ .md-button }
[Continue to Policy Gradients (REINFORCE) →](policy_based.md){ .md-button .md-button--primary }


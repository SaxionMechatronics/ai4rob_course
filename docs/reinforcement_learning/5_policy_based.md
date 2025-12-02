# Policy Gradients: REINFORCE Algorithm

## From Values to Policies

In Q-Learning, we learned to estimate the value of actions and then derived a policy from those values. But what if we **directly learn the policy** instead?

**Policy gradient methods** do exactly this: they parameterize the policy and optimize it using gradient ascent to maximize expected return.

## The Policy Gradient Approach

### Parameterized Policy

Instead of learning Q-values, we directly parameterize the policy with parameters \( \theta \):

\[
\pi_\theta(a | s)
\]

**Examples:**
- **Neural network**: Input state, output action probabilities
- **Linear model**: \( \pi_\theta(a|s) = \text{softmax}(\theta^T \phi(s)) \)
- **Gaussian policy**: \( \pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)) \)

### The Objective

We want to find parameters \( \theta \) that maximize expected return:

\[
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [G(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
\]

Where \( \tau = (s_0, a_0, r_0, s_1, a_1, r_1, ...) \) is a trajectory sampled by following policy \( \pi_\theta \).

**Intuition:** We're directly optimizing what we care about—the total reward!

### Gradient Ascent

To maximize \( J(\theta) \), we use gradient ascent:

\[
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
\]

Move parameters in the direction that increases expected return.

**The challenge:** How do we compute \( \nabla_\theta J(\theta) \)?

## The Policy Gradient Theorem

The **policy gradient theorem** gives us a practical way to compute the gradient:

\[
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) G_t \right]
\]

Where \( G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k \) is the return from time \( t \).

**What this means:**
- Sample trajectories using current policy \( \pi_\theta \)
- For each action taken, compute its log-probability gradient
- Weight by the return obtained from that point onward
- Average over many trajectories

**Intuitive interpretation:**
- If action \( a_t \) led to high return \( G_t \): increase \( \pi_\theta(a_t | s_t) \)
- If action \( a_t \) led to low return \( G_t \): decrease \( \pi_\theta(a_t | s_t) \)

## REINFORCE: The Monte Carlo Policy Gradient Algorithm

**REINFORCE** (Williams, 1992) is the classic policy gradient algorithm. It's beautifully simple!

### The Algorithm

```
Initialize policy parameters θ randomly

For each episode:
    1. Generate trajectory τ = (s₀, a₀, r₁, s₁, ..., sT)
       by following πθ
    
    2. For each timestep t in the episode:
        Compute return: Gt = Σ(k=t to T) γ^(k-t) * r_k
        
        Compute gradient: ∇θ log πθ(at | st)
        
        Update: θ ← θ + α * Gt * ∇θ log πθ(at | st)
```

### Why This Works: Intuition

Imagine training a robot to navigate:

**Episode 1:** Robot wanders randomly, finds goal by luck
- Total return: G = 50 (pretty good!)
- Actions taken are reinforced (made more likely)

**Episode 2:** Robot goes wrong direction, fails
- Total return: G = -10 (bad!)
- Actions taken are suppressed (made less likely)

Over many episodes, good actions become more probable, bad actions less probable.

### The Log-Probability Trick

Why do we use \( \nabla_\theta \log \pi_\theta(a|s) \) instead of \( \nabla_\theta \pi_\theta(a|s) \)?

**Mathematical reason:** The policy gradient theorem gives us this form.

**Practical reason:** It's easier to compute!

For a softmax policy:
\[
\pi_\theta(a|s) = \frac{e^{\theta^T \phi(s, a)}}{\sum_{a'} e^{\theta^T \phi(s, a')}}
\]

\[
\nabla_\theta \log \pi_\theta(a|s) = \phi(s, a) - \mathbb{E}_{a' \sim \pi_\theta}[\phi(s, a')]
\]

The gradient is just the feature vector minus its expectation!

For neural networks with automatic differentiation (PyTorch, TensorFlow), computing log-probability gradients is straightforward.

## REINFORCE with Baseline

### The High Variance Problem

Pure REINFORCE has a critical issue: **very high variance** in gradient estimates.

**Why?** Returns \( G_t \) can vary wildly between episodes, even for similar actions.

**Example:**
```
Episode 1: Gt = 100  → Big update!
Episode 2: Gt = 95   → Big update in slightly different direction
Episode 3: Gt = 5    → Big negative update
Episode 4: Gt = 90   → Big positive update again
```

The agent gets conflicting signals, making learning slow and unstable.

### Introducing a Baseline

We can subtract a **baseline** \( b(s_t) \) from the return without changing the expected gradient:

\[
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) (G_t - b(s_t)) \right]
\]

**Why this doesn't introduce bias:** The expectation of \( \nabla_\theta \log \pi_\theta(a|s) \) is zero, so adding/subtracting baselines doesn't change the expected gradient.

**Why it reduces variance:** By centering the returns around zero, we reduce the magnitude of gradient updates.

### Choosing a Baseline

**Best baseline:** The state-value function \( V(s_t) \)!

\[
G_t - V(s_t) = A_t
\]

This is called the **advantage** \( A_t \): how much better was this action compared to the average?

**Interpretation:**
- \( A_t > 0 \): Action was better than expected → increase probability
- \( A_t < 0 \): Action was worse than expected → decrease probability
- \( A_t = 0 \): Action was as expected → no change

**Practical implementation:** Learn \( V(s) \) using a separate value network (this leads to Actor-Critic methods!)

### REINFORCE with Baseline Algorithm

```
Initialize policy parameters θ and value function parameters w

For each episode:
    Generate trajectory τ = (s₀, a₀, r₁, s₁, ..., sT) using πθ
    
    For each timestep t:
        Compute return: Gt = Σ(k=t to T) γ^(k-t) * r_k
        
        Compute advantage: At = Gt - V_w(st)
        
        Update policy: θ ← θ + α * At * ∇θ log πθ(at | st)
        
        Update value: w ← w + β * (Gt - V_w(st)) * ∇w V_w(st)
```

## Policy Representations

### Discrete Actions: Softmax Policy

For discrete action spaces, use a **softmax (categorical) policy**:

\[
\pi_\theta(a | s) = \frac{e^{f_\theta(s, a)}}{\sum_{a'} e^{f_\theta(s, a')}}
\]

Where \( f_\theta(s, a) \) is a neural network that outputs action logits.

**Network architecture:**
```
State s → [Neural Network] → [Logits for each action] → Softmax → Probabilities
```

**Sampling:**
```python
logits = policy_network(state)
action_probs = softmax(logits)
action = categorical_sample(action_probs)
```

**Log-probability:**
```python
log_prob = log_softmax(logits)[action]
```

### Continuous Actions: Gaussian Policy

For continuous action spaces, use a **Gaussian policy**:

\[
\pi_\theta(a | s) = \mathcal{N}(a | \mu_\theta(s), \sigma_\theta(s))
\]

Where the neural network outputs mean \( \mu_\theta(s) \) and standard deviation \( \sigma_\theta(s) \).

**Network architecture:**
```
State s → [Neural Network] → [μ₁, μ₂, ..., μn, σ₁, σ₂, ..., σn]
```

**Sampling:**
```python
mu, sigma = policy_network(state)
action = mu + sigma * random_normal()
```

**Log-probability:**
```python
log_prob = -0.5 * ((action - mu) / sigma)² - log(sigma) - 0.5 * log(2π)
```

**Practical tip:** Often use a diagonal Gaussian (independent dimensions) for simplicity.

## Simple Python Implementation

Here's a minimal REINFORCE implementation using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        logits = self.network(state)
        return logits

class REINFORCE:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        
    def select_action(self, state):
        """Select action and compute log probability"""
        state = torch.FloatTensor(state)
        logits = self.policy(state)
        
        # Sample from categorical distribution
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def compute_returns(self, rewards):
        """Compute discounted returns for each timestep"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update(self, log_probs, rewards):
        """Update policy using REINFORCE"""
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # Compute policy gradient loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        # Perform gradient ascent
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

# Training loop
agent = REINFORCE(state_dim=4, action_dim=2)

for episode in range(1000):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    
    # Collect trajectory
    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state
    
    # Update policy
    agent.update(log_probs, rewards)
    
    print(f"Episode {episode}: Total Reward = {sum(rewards)}")
```

## REINFORCE with Baseline Implementation

Adding a value function baseline:

```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state).squeeze()

class REINFORCEWithBaseline:
    def __init__(self, state_dim, action_dim, lr_policy=1e-3, lr_value=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr_value)
        self.gamma = gamma
    
    def update(self, states, log_probs, rewards):
        """Update policy and value function"""
        # Compute returns
        returns = self.compute_returns(rewards)
        
        states = torch.FloatTensor(states)
        returns = torch.FloatTensor(returns)
        
        # Compute value predictions
        values = self.value(states)
        
        # Compute advantages
        advantages = returns - values.detach()  # Don't backprop through value for policy
        
        # Update policy
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        
        self.policy_optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value function (minimize MSE)
        value_loss = ((values - returns) ** 2).mean()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
```

## Advantages of Policy Gradients

✅ **Continuous actions**: Naturally handles continuous action spaces

✅ **Stochastic policies**: Can learn optimal stochastic policies

✅ **Convergence**: Better convergence properties than value-based methods

✅ **High-dimensional actions**: Doesn't need argmax over action space

✅ **Stable**: Generally more stable than Q-learning with function approximation

## Disadvantages of Policy Gradients

❌ **Sample inefficient**: Requires many episodes to estimate gradients accurately

❌ **High variance**: Gradient estimates can be very noisy

❌ **On-policy**: Must collect new data after each policy update (expensive!)

❌ **Local optima**: Gradient ascent can get stuck in local maxima

❌ **Slow convergence**: Can take many iterations to learn

## Variance Reduction Techniques

Beyond baselines, several techniques reduce gradient variance:

### 1. Reward-to-Go

Instead of using full episode return \( G_0 \), use **reward-to-go** \( G_t \):

\[
G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k
\]

**Why it helps:** Action at time \( t \) doesn't affect rewards before time \( t \), so we shouldn't credit it for those rewards.

### 2. Generalized Advantage Estimation (GAE)

Combine multiple \( n \)-step returns with exponential weighting:

\[
A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
\]

Where \( \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \) is the TD error.

**Parameter \( \lambda \):** Controls bias-variance tradeoff
- \( \lambda = 0 \): Low variance, high bias (like TD)
- \( \lambda = 1 \): High variance, low bias (like Monte Carlo)

### 3. Entropy Regularization

Add entropy bonus to encourage exploration:

\[
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)] + \beta \mathcal{H}(\pi_\theta)
\]

Where \( \mathcal{H}(\pi_\theta) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s) \) is the policy entropy.

**Effect:** Prevents premature convergence to deterministic policy, encourages exploration.

## Modern Policy Gradient Methods

REINFORCE is the foundation, but modern methods add improvements:

### PPO (Proximal Policy Optimization)

**Key idea:** Limit how much the policy can change in one update.

**Clipped objective:**
\[
L(\theta) = \mathbb{E} \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]
\]

Where \( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \) is the probability ratio.

**Why it works:** Prevents destructively large updates, more stable training.

### TRPO (Trust Region Policy Optimization)

**Key idea:** Constrain policy updates to stay within a "trust region."

**Constraint:**
\[
\mathbb{E}[D_{KL}(\pi_{\theta_{\text{old}}} || \pi_\theta)] \leq \delta
\]

Ensures new policy isn't too different from old policy.

**Why it works:** Guarantees monotonic improvement, very stable.

### A3C/A2C (Advantage Actor-Critic)

**Key idea:** Use value function as baseline, update after every step (not full episodes).

**Benefits:**
- Lower variance (value function baseline)
- More sample efficient (use bootstrapping like TD)
- Can train in parallel (A3C)

We'll cover actor-critic methods in detail in the next section!

## Comparing REINFORCE to Q-Learning

| Aspect | Q-Learning | REINFORCE |
|--------|------------|-----------|
| **What we learn** | Q(s, a) values | Policy πθ directly |
| **Update frequency** | Every step | Every episode |
| **Sample efficiency** | More efficient | Less efficient |
| **Action space** | Discrete only | Discrete or continuous |
| **Variance** | Lower | Higher |
| **Bias** | Biased (with function approx) | Unbiased |
| **Convergence** | Can diverge | More stable |
| **Off-policy** | Yes | No (by default) |

**When to use REINFORCE over Q-Learning:**
- Continuous action spaces (robot control)
- Stochastic policies needed
- Stability more important than sample efficiency

## Practical Tips for Training

### 1. Normalize Returns

```python
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

Helps stabilize training by keeping gradients in reasonable range.

### 2. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
```

Prevents exploding gradients.

### 3. Learning Rate Scheduling

Start with higher learning rate, decay over time:
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
```

### 4. Early Stopping

Monitor validation performance and stop if not improving.

### 5. Multiple Random Seeds

Train with different random seeds, report mean and std of performance.

## Summary

**REINFORCE in a nutshell:**

1. **Sample trajectories**: Follow current policy to collect episodes
2. **Compute returns**: Calculate discounted reward-to-go for each timestep  
3. **Compute advantages**: Subtract baseline (optional but recommended)
4. **Update policy**: Gradient ascent weighted by advantages

**Key equation:**
\[
\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) (G_t - b(s_t))
\]

**Strengths:** Simple, direct optimization, handles continuous actions, stable convergence

**Weaknesses:** Sample inefficient, high variance, on-policy

**Modern extensions:** PPO, TRPO, A3C (address sample efficiency and variance)

Now let's see REINFORCE in action with a hands-on implementation!

---

[← Back to Q-Learning](4_value_based.md){ .md-button }
[Continue to Practical Tutorial →](6_practical_tutorial.md){ .md-button .md-button--primary }


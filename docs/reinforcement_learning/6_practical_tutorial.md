# Putting It All Together: Practical RL Implementation

## Overview

In this tutorial, we'll implement REINFORCE from scratch and compare it with PPO from Stable-Baselines3 (SB3). You'll see:

1. How to implement REINFORCE algorithm step-by-step
2. How to use Gymnasium (OpenAI Gym) environments
3. How to train and evaluate your agent
4. How your implementation compares to state-of-the-art (PPO)
5. Practical tips for debugging and improving performance

## Environment: CartPole-v1

We'll use **CartPole-v1**, a classic control problem:

**Goal:** Balance a pole on a moving cart by applying left/right forces.

**Observations:** 
- Cart position: \([-4.8, 4.8]\)
- Cart velocity: \([-\infty, \infty]\)
- Pole angle: \([-0.418, 0.418]\) rad (≈24°)
- Pole angular velocity: \([-\infty, \infty]\)

**Actions:**
- 0: Push cart to the left
- 1: Push cart to the right

**Rewards:**
- +1 for every timestep the pole remains upright
- Episode ends when pole falls > 12° or cart moves > 2.4 units from center

**Success criterion:** Average reward of 475 over 100 episodes (max is 500)

## Part 1: REINFORCE from Scratch

### Step 1: Setup and Imports

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

### Step 2: Policy Network

```python
class PolicyNetwork(nn.Module):
    """
    Neural network that outputs action probabilities
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        """
        Forward pass: state -> action logits
        """
        return self.network(state)
    
    def get_action(self, state):
        """
        Sample action from policy and return log probability
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.forward(state)
        
        # Create categorical distribution
        dist = Categorical(logits=logits)
        
        # Sample action
        action = dist.sample()
        
        # Get log probability for policy gradient
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
```

### Step 3: REINFORCE Agent

```python
class REINFORCEAgent:
    """
    REINFORCE algorithm implementation
    """
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99):
        self.gamma = gamma
        
        # Initialize policy network
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Storage for episode data
        self.reset_episode()
        
    def reset_episode(self):
        """Clear episode storage"""
        self.log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        """Select action using current policy"""
        action, log_prob = self.policy.get_action(state)
        self.log_probs.append(log_prob)
        return action
    
    def store_reward(self, reward):
        """Store reward for current timestep"""
        self.rewards.append(reward)
    
    def compute_returns(self):
        """
        Compute discounted returns (reward-to-go) for each timestep
        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        """
        returns = []
        G = 0
        
        # Compute returns in reverse order
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensor
        returns = torch.FloatTensor(returns)
        
        # Normalize returns for stability (optional but recommended)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update_policy(self):
        """
        Update policy using REINFORCE algorithm
        """
        # Compute returns
        returns = self.compute_returns()
        
        # Compute policy gradient loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            # Negative because we're doing gradient ascent
            policy_loss.append(-log_prob * G)
        
        # Sum all losses
        policy_loss = torch.stack(policy_loss).sum()
        
        # Perform gradient ascent step
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Clear episode data
        self.reset_episode()
        
        return policy_loss.item()
```

### Step 4: Training Loop

```python
def train_reinforce(env_name='CartPole-v1', 
                   num_episodes=1000, 
                   learning_rate=1e-2,
                   gamma=0.99,
                   print_every=50):
    """
    Train REINFORCE agent
    """
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = REINFORCEAgent(state_dim, action_dim, learning_rate, gamma)
    
    # Training tracking
    episode_rewards = []
    episode_lengths = []
    running_reward = deque(maxlen=100)
    
    print(f"Training REINFORCE on {env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print("-" * 50)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        # Collect one episode
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store reward
            agent.store_reward(reward)
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Update policy after episode
        loss = agent.update_policy()
        
        # Track statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        running_reward.append(episode_reward)
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(running_reward)
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Last Reward: {episode_reward:.2f} | "
                  f"Loss: {loss:.4f}")
            
            # Check if solved
            if avg_reward >= 475.0:
                print(f"\n🎉 Solved in {episode + 1} episodes! "
                      f"Average reward: {avg_reward:.2f}")
                break
    
    env.close()
    
    return agent, episode_rewards, episode_lengths
```

### Step 5: Evaluation Function

```python
def evaluate_agent(agent, env_name='CartPole-v1', num_episodes=100, render=False):
    """
    Evaluate trained agent
    """
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action (greedy, no exploration)
            with torch.no_grad():
                action, _ = agent.policy.get_action(state)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
    
    env.close()
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return episode_rewards
```

### Step 6: Visualization

```python
def plot_training_progress(rewards, window=100):
    """
    Plot training progress
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot raw rewards
    ax1.plot(rewards, alpha=0.3, label='Raw')
    
    # Plot moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 
                label=f'{window}-episode moving average')
    
    ax1.axhline(y=475, color='r', linestyle='--', label='Solved threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot distribution of recent rewards
    recent_rewards = rewards[-100:] if len(rewards) > 100 else rewards
    ax2.hist(recent_rewards, bins=20, edgecolor='black')
    ax2.axvline(x=np.mean(recent_rewards), color='r', linestyle='--', 
               label=f'Mean: {np.mean(recent_rewards):.2f}')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Recent Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

### Step 7: Run Training

```python
# Train the agent
agent, rewards, lengths = train_reinforce(
    env_name='CartPole-v1',
    num_episodes=1000,
    learning_rate=1e-2,
    gamma=0.99
)

# Plot results
plot_training_progress(rewards)

# Evaluate the trained agent
eval_rewards = evaluate_agent(agent, num_episodes=100)
```

## Part 2: PPO from Stable-Baselines3

Now let's compare with PPO, a state-of-the-art algorithm:

### Setup

```python
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import time
```

### Train PPO

```python
def train_ppo(env_name='CartPole-v1', 
              total_timesteps=100000,
              learning_rate=3e-4):
    """
    Train PPO agent using Stable-Baselines3
    """
    print(f"\nTraining PPO on {env_name}")
    print("-" * 50)
    
    # Create environment
    env = gym.make(env_name)
    
    # Create PPO agent
    model = PPO(
        'MlpPolicy',  # Multi-layer perceptron policy
        env,
        learning_rate=learning_rate,
        n_steps=2048,  # Collect 2048 steps before each update
        batch_size=64,
        n_epochs=10,  # Number of epochs per update
        gamma=0.99,
        gae_lambda=0.95,  # For advantage estimation
        clip_range=0.2,  # PPO clipping parameter
        verbose=1
    )
    
    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return model

# Train PPO
ppo_model = train_ppo(total_timesteps=100000)
```

### Evaluate PPO

```python
# Evaluate PPO
env = gym.make('CartPole-v1')
mean_reward, std_reward = evaluate_policy(
    ppo_model, 
    env, 
    n_eval_episodes=100
)

print(f"\nPPO Evaluation over 100 episodes:")
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()
```

## Part 3: Detailed Comparison

### Comparison Function

```python
def compare_algorithms():
    """
    Train both algorithms and compare
    """
    print("=" * 60)
    print("REINFORCE vs PPO Comparison on CartPole-v1")
    print("=" * 60)
    
    # Train REINFORCE
    print("\n1. Training REINFORCE...")
    reinforce_start = time.time()
    reinforce_agent, reinforce_rewards, _ = train_reinforce(
        num_episodes=1000,
        learning_rate=1e-2,
        print_every=100
    )
    reinforce_time = time.time() - reinforce_start
    
    # Evaluate REINFORCE
    reinforce_eval = evaluate_agent(reinforce_agent, num_episodes=100)
    reinforce_mean = np.mean(reinforce_eval)
    reinforce_std = np.std(reinforce_eval)
    
    # Train PPO
    print("\n2. Training PPO...")
    ppo_start = time.time()
    ppo_model = train_ppo(total_timesteps=100000)
    ppo_time = time.time() - ppo_start
    
    # Evaluate PPO
    env = gym.make('CartPole-v1')
    ppo_mean, ppo_std = evaluate_policy(ppo_model, env, n_eval_episodes=100)
    env.close()
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\nPerformance:")
    print(f"  REINFORCE: {reinforce_mean:.2f} ± {reinforce_std:.2f}")
    print(f"  PPO:       {ppo_mean:.2f} ± {ppo_std:.2f}")
    
    print("\nTraining Time:")
    print(f"  REINFORCE: {reinforce_time:.2f} seconds")
    print(f"  PPO:       {ppo_time:.2f} seconds")
    
    print("\nSample Efficiency:")
    reinforce_episodes = len(reinforce_rewards)
    reinforce_steps = sum([len(r) for r in reinforce_rewards]) if isinstance(reinforce_rewards[0], list) else reinforce_episodes * 200  # Approximate
    print(f"  REINFORCE: ~{reinforce_episodes} episodes")
    print(f"  PPO:       ~100,000 timesteps")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training curves
    axes[0].plot(reinforce_rewards, label='REINFORCE', alpha=0.7)
    axes[0].axhline(y=475, color='r', linestyle='--', label='Solved')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Evaluation comparison
    algorithms = ['REINFORCE', 'PPO']
    means = [reinforce_mean, ppo_mean]
    stds = [reinforce_std, ppo_std]
    
    axes[1].bar(algorithms, means, yerr=stds, capsize=10, 
               color=['blue', 'orange'], alpha=0.7)
    axes[1].axhline(y=475, color='r', linestyle='--', label='Solved')
    axes[1].set_ylabel('Mean Reward')
    axes[1].set_title('Evaluation Performance (100 episodes)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'reinforce': {
            'mean': reinforce_mean,
            'std': reinforce_std,
            'time': reinforce_time,
            'rewards': reinforce_rewards
        },
        'ppo': {
            'mean': ppo_mean,
            'std': ppo_std,
            'time': ppo_time
        }
    }

# Run comparison
results = compare_algorithms()
```

## Expected Results and Analysis

### REINFORCE Performance

**Typical outcomes:**
- **Convergence**: 300-600 episodes to solve (avg reward ≥ 475)
- **Stability**: Some variance, may occasionally drop performance
- **Final performance**: 450-500 mean reward
- **Training time**: 1-3 minutes on CPU

**Characteristics:**
- High variance in early training
- Gradual, sometimes unsteady improvement
- Sensitive to hyperparameters (learning rate, gamma)
- Simple and interpretable

### PPO Performance

**Typical outcomes:**
- **Convergence**: Solves within 50,000-100,000 timesteps
- **Stability**: Very stable training curve
- **Final performance**: 495-500 mean reward (near optimal)
- **Training time**: 30-60 seconds on CPU

**Characteristics:**
- Smooth, consistent improvement
- Very stable due to clipped updates
- More sample efficient than REINFORCE
- More complex implementation

### Key Differences

| Aspect | REINFORCE | PPO |
|--------|-----------|-----|
| **Sample Efficiency** | Lower | Higher |
| **Stability** | Moderate | High |
| **Variance** | High | Low |
| **Complexity** | Simple | Complex |
| **Convergence Speed** | Slower | Faster |
| **Final Performance** | Good | Excellent |
| **Hyperparameter Sensitivity** | High | Moderate |

## Part 4: Debugging and Improvements

### Common Issues with REINFORCE

#### Issue 1: Not Learning / Random Performance

**Symptoms:** Rewards stay around 20-30, no improvement

**Possible causes:**
- Learning rate too high or too low
- No return normalization
- Incorrect gradient computation

**Solutions:**
```python
# Try different learning rates
learning_rates = [1e-4, 1e-3, 1e-2]

# Ensure return normalization
returns = (returns - returns.mean()) / (returns.std() + 1e-8)

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
```

#### Issue 2: Unstable Learning

**Symptoms:** Performance improves then suddenly drops

**Solutions:**
```python
# Use smaller learning rate
learning_rate = 1e-3

# Add entropy bonus for exploration
entropy = dist.entropy().mean()
loss = policy_loss - 0.01 * entropy

# Increase gamma for longer-term thinking
gamma = 0.99 or 0.995
```

#### Issue 3: Slow Convergence

**Solutions:**
```python
# Use baseline (value function)
advantage = returns - baseline

# Better network architecture
hidden_dim = 256  # Larger network

# Use adaptive learning rate
optimizer = optim.Adam(policy.parameters(), lr=1e-2, eps=1e-5)
```

### Hyperparameter Tuning Guide

```python
def hyperparameter_search():
    """
    Grid search over hyperparameters
    """
    learning_rates = [1e-3, 3e-3, 1e-2]
    gammas = [0.95, 0.99, 0.995]
    hidden_dims = [64, 128, 256]
    
    best_score = 0
    best_params = {}
    
    for lr in learning_rates:
        for gamma in gammas:
            for hidden in hidden_dims:
                print(f"\nTrying lr={lr}, gamma={gamma}, hidden={hidden}")
                
                # Train with these params
                # ... (training code)
                
                # Evaluate
                score = evaluate_agent(agent)
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'lr': lr,
                        'gamma': gamma,
                        'hidden': hidden
                    }
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score}")
    
    return best_params
```

## Part 5: Extensions and Next Steps

### Extension 1: REINFORCE with Baseline

```python
class REINFORCEWithBaseline(REINFORCEAgent):
    """
    REINFORCE with value function baseline
    """
    def __init__(self, state_dim, action_dim, lr_policy=1e-3, lr_value=1e-3, gamma=0.99):
        super().__init__(state_dim, action_dim, lr_policy, gamma)
        
        # Add value network
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr_value)
        
        self.states = []
    
    def select_action(self, state):
        self.states.append(state)
        return super().select_action(state)
    
    def update_policy(self):
        returns = self.compute_returns()
        states = torch.FloatTensor(np.array(self.states))
        
        # Compute values
        values = self.value_network(states).squeeze()
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Update policy
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Update value function
        value_loss = ((values - returns) ** 2).mean()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Reset
        self.reset_episode()
        self.states = []
        
        return policy_loss.item(), value_loss.item()
```

### Extension 2: Try Other Environments

```python
# More challenging environments
environments = [
    'CartPole-v1',      # Solved: 475
    'Acrobot-v1',       # Solved: -100
    'LunarLander-v2',   # Solved: 200
    'MountainCar-v0'    # Solved: -110
]

for env_name in environments:
    print(f"\n{'='*60}")
    print(f"Training on {env_name}")
    print(f"{'='*60}")
    
    agent, rewards, _ = train_reinforce(
        env_name=env_name,
        num_episodes=2000
    )
    
    plot_training_progress(rewards)
```

### Extension 3: Continuous Actions

```python
from torch.distributions import Normal

class ContinuousPolicyNetwork(nn.Module):
    """
    Policy network for continuous actions (Gaussian policy)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        x = self.shared(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.forward(state)
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze().numpy(), log_prob

# Use this for continuous control tasks like Pendulum-v1, HalfCheetah, etc.
```

## Summary

In this tutorial, you've:

✅ Implemented REINFORCE from scratch with detailed explanations

✅ Learned how to train and evaluate RL agents

✅ Compared your implementation with state-of-the-art PPO

✅ Understood the tradeoffs between different algorithms

✅ Learned debugging techniques and hyperparameter tuning

✅ Explored extensions like baselines and continuous actions

### Key Takeaways

1. **REINFORCE is simple** but effective for basic problems
2. **Modern algorithms (PPO)** are more sample efficient and stable
3. **Implementation details matter**: normalization, clipping, learning rates
4. **Start simple**: Get basic version working before adding complexity
5. **Evaluation is critical**: Don't just look at training curves

### Next Steps

1. **Try harder environments**: LunarLander, MuJoCo tasks
2. **Implement Actor-Critic**: A2C or A3C
3. **Add GAE**: Generalized Advantage Estimation
4. **Try PPO from scratch**: Implement the clipping objective
5. **Real robotics**: Sim-to-real transfer with domain randomization

## Complete Code

The complete, runnable code is available in the accompanying Jupyter notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

*Link to notebook will be added here*

## References

- **REINFORCE**: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
- **PPO**: Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
- **Stable-Baselines3**: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)
- **Gymnasium**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

---

[← Back to Policy Gradients](5_policy_based.md){ .md-button }
[Back to RL Module Home](index.md){ .md-button .md-button--primary }


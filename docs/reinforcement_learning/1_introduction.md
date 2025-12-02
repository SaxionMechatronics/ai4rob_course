# Introduction to Reinforcement Learning

## What is Reinforcement Learning?

Imagine teaching a dog a new trick. You don't explain the physics of jumping or write out step-by-step instructions. Instead, you reward the dog when it does something right and provide no reward (or a gentle correction) when it doesn't. Over time, through trial and error, the dog learns which actions lead to treats and praise.

This is exactly how **Reinforcement Learning (RL)** works for robots and AI systems!

### The Core Idea

Reinforcement Learning is about learning through **interaction** and **feedback**. Unlike supervised learning where we tell the system exactly what to do (like showing labeled examples), in RL:

- An **agent** (robot, drone, game AI) takes actions in an **environment** (physical world, simulation)
- The environment gives **feedback** in the form of rewards or penalties
- The agent learns from experience which actions lead to better outcomes
- Over time, the agent discovers strategies (called **policies**) that maximize its total reward

Think of it like learning to ride a bike:
- You try different balancing strategies (actions)
- Staying upright feels good, falling hurts (rewards/penalties)
- You don't need someone to tell you the exact muscle movements
- Through practice, you develop an intuition for how to balance

### Why is RL Important for Robotics?

Traditional robot control often requires:
- Expert knowledge to program every behavior
- Extensive modeling of the environment physics
- Manual tuning of parameters for different scenarios
- Difficulty adapting to new or unexpected situations

RL enables robots to:
- **Learn from experience** rather than explicit programming
- **Discover novel solutions** that humans might not think of
- **Adapt to changing environments** and unexpected situations
- **Optimize complex behaviors** that are hard to engineer manually

## Real-World Examples: RL in Action

Let's look at some impressive examples where RL has been preferred over traditional control methods:

### 🐕 Boston Dynamics Spot

<div class="admonition example">
<p class="admonition-title">Spot's Adaptive Locomotion</p>

Boston Dynamics uses RL to teach Spot to walk on challenging terrain. Traditional approaches would require:
- Detailed terrain mapping and classification
- Pre-programmed gaits for every surface type
- Complex state machines for transitions
- Extensive manual tuning

**With RL:**
- Spot learns to adapt its gait by trial and error in simulation
- Discovers robust walking strategies that work on unseen terrains
- Automatically learns recovery behaviors when it slips or trips
- Transfers learned skills from simulation to the real robot

**Why RL over traditional control?**
- Too many terrain variations to program manually
- Novel recovery strategies emerge that engineers didn't anticipate
- Adaptability to completely new environments without reprogramming
</div>

### 🚁 Autonomous Drone Racing

<div class="admonition example">
<p class="admonition-title">High-Speed Drone Navigation</p>

Researchers have trained drones to race through obstacle courses at speeds exceeding 40 mph, beating human pilots!

**Traditional approach challenges:**
- Requires perfect sensing and precise trajectory planning
- Computational delays in processing and replanning
- Difficult to handle aggressive maneuvers and aerodynamic effects
- Brittle to sensor noise and unexpected obstacles

**RL advantages:**
- Learns end-to-end control from camera images to motor commands
- Develops reactive behaviors that don't require explicit planning
- Discovers aggressive maneuvers through exploration
- Naturally handles partial observability and sensor uncertainty
- Achieves super-human performance through extensive simulated practice

**Key insight:** The RL-trained policy makes decisions in milliseconds based on pattern recognition rather than slow deliberative planning, similar to how expert human pilots develop intuition.
</div>

### 🦙 ANYmal (ETH Zurich)

<div class="admonition example">
<p class="admonition-title">Quadruped Robot Traversing Rough Terrain</p>

ETH Zurich's ANYmal is a four-legged robot that can climb stairs, traverse rubble, and recover from slips using RL.

**Why not traditional control?**
- Traditional model-based control requires:
  - Accurate terrain geometry (not always available)
  - Perfect knowledge of robot dynamics
  - Conservative behaviors to ensure stability
  - Separate controllers for different scenarios

**RL approach benefits:**
- Learns directly from proprioceptive sensors (joint positions, torques, IMU)
- Doesn't need explicit terrain geometry
- Discovers dynamic gaits that traditional methods might consider "unsafe"
- Single learned policy handles walking, trotting, and recovery
- Robust to model uncertainty and external disturbances

**Result:** ANYmal can walk blindly (without vision) on very rough terrain by learning to predict terrain properties from how its legs interact with the ground.
</div>

### 🦾 Robotic Manipulation

<div class="admonition example">
<p class="admonition-title">Dexterous In-Hand Manipulation</p>

OpenAI trained a robotic hand to solve a Rubik's cube using RL, demonstrating human-level dexterity.

**Traditional approach limitations:**
- Extremely difficult to model contact dynamics accurately
- Hand-crafted control policies are brittle and task-specific
- Requires precise sensing and perfect calibration
- Fails when objects slip or unexpected perturbations occur

**RL advantages:**
- Learns robust manipulation through millions of simulated attempts
- Develops recovery strategies for when objects slip
- Discovers creative fingering strategies
- Handles visual ambiguity and partial observability
- Shows emergent behaviors like flipping the cube in creative ways

**Critical factor:** Domain randomization in simulation (varying physics parameters, visual appearance, etc.) allows the learned policy to be robust enough to transfer to the real world despite sim-to-real gap.
</div>

### 🎮 Game AI: AlphaGo and Beyond

<div class="admonition example">
<p class="admonition-title">Mastering Complex Strategic Games</p>

While not robotics, AlphaGo's achievement in mastering Go demonstrates RL's capability to solve problems previously thought to require human intuition.

**Why RL?**
- Search space is too large for brute-force approaches (more positions than atoms in the universe!)
- Human expert knowledge is incomplete and biased
- Requires long-term strategic thinking and pattern recognition

**What this means for robotics:**
- RL can solve problems where the optimal solution isn't known
- Can discover strategies that exceed human expertise
- Learns hierarchical representations and abstract concepts
- Demonstrates that RL scales to extremely complex decision-making
</div>

## When Should You Use RL for Robotics?

RL is particularly powerful when:

✅ **The optimal behavior is unknown** - You know what you want to achieve, but not exactly how

✅ **The environment is complex or stochastic** - Too many variables to model accurately

✅ **You can simulate** - RL needs lots of experience; simulation makes this feasible

✅ **Adaptability is crucial** - The robot needs to handle varied, unpredictable situations

✅ **Trial-and-error is safe** - At least in simulation, the robot can make mistakes

❌ **Avoid RL when:**
- You have a well-understood problem with a known solution
- Safety is critical and you cannot guarantee safe exploration
- You cannot simulate effectively and real-world data is scarce
- Simple control methods would suffice
- Interpretability and guarantees are essential

## The Learning Process: A High-Level View

Here's how an RL system learns, using a robot learning to walk as an example:

1. **Start with random behavior**: The robot tries random motor commands
   - It immediately falls over! *Reward: -10*

2. **Try something different**: Through exploration, it finds that certain joint angles keep it upright longer
   - It stands for 2 seconds before falling. *Reward: +2*

3. **Build on success**: It remembers that these joint configurations were good and tries variations
   - It takes a wobbly step forward. *Reward: +5*

4. **Discover better strategies**: Through thousands of attempts, it finds that shifting weight to one side allows stepping with the other leg
   - It takes several steps before falling. *Reward: +15*

5. **Refine and optimize**: Continue learning subtle adjustments to balance, speed, and energy efficiency
   - Smooth, stable walking. *Reward: +100*

6. **Handle edge cases**: Learn recovery behaviors when pushed or walking on slopes
   - Successfully recovers from perturbations. *Reward: +50*

The key insight: The robot never explicitly learned "how" to walk in human terms. It discovered patterns of actions that led to high rewards through systematic trial and error.

## The Three Key Questions

Every RL problem can be understood through three questions:

1. **What can the agent observe?** (Observations/State)
   - Camera images? Joint angles? GPS coordinates?

2. **What can the agent do?** (Actions)
   - Motor commands? High-level waypoints? Discrete choices?

3. **What does the agent want to achieve?** (Rewards)
   - Reach a goal? Move quickly? Use less energy? Stay balanced?

Designing these three components well is often the key to successful RL applications!

## Coming Up Next

Now that you understand the intuition and real-world applications of RL, we'll dive deeper into:

- **Markov Decision Processes (MDPs)**: The mathematical framework that formalizes these concepts
- **Value-based vs Policy-based methods**: Two different philosophical approaches to learning
- **Specific algorithms**: Q-Learning, REINFORCE, PPO, and more
- **Hands-on implementation**: Building your own RL agent from scratch

Ready to dive deeper? Let's start with the formal framework that makes all of this work!

---

## Check your understanding
[Quiz 1](1_quiz.md){ .md-button }

[Continue to Markov Decision Processes →](2_mdp.md){ .md-button .md-button--primary }

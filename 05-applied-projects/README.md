# 🎮 Reinforcement Learning

An exploration of how agents learn to make decisions through interaction with environments. This section covers fundamental RL algorithms, from value-based methods to policy gradient approaches, demonstrating how machines learn optimal behavior through trial and error.

## 📋 Overview

Reinforcement Learning enables agents to learn optimal strategies by receiving rewards and penalties from their environment. Unlike supervised learning, RL agents must discover successful behaviors through exploration, balancing immediate rewards with long-term goals. This section implements core RL algorithms from first principles.

## 📂 Contents

### Value-Based Methods

#### 🎯 **Q-Learning Implementation**
*File: `q-learning-implementation.ipynb`*

The foundational algorithm for learning action-value functions:
- Markov Decision Processes (MDPs): states, actions, rewards, transitions
- Q-function: estimating expected cumulative reward for state-action pairs
- Bellman equation and temporal difference learning
- Epsilon-greedy exploration vs. exploitation trade-off
- Q-table updates and convergence properties
- Grid world and classic control environments
- Hyperparameter tuning: learning rate, discount factor, exploration rate
- Visualization of learned policies and value functions

**Key Concepts:** Value iteration, TD learning, exploration-exploitation, convergence guarantees

---

### Policy-Based Methods

#### 🚀 **Policy Gradient Methods**
*File: `policy-gradient-methods.ipynb`*

Learning policies directly through gradient ascent on expected rewards:
- Policy representation: parameterized action distributions
- REINFORCE algorithm: Monte Carlo policy gradient
- Log-derivative trick and score function estimator
- Baseline reduction: reducing variance with value function baselines
- Actor-Critic methods: combining policy and value learning
- Advantage estimation: A(s,a) = Q(s,a) - V(s)
- Continuous action spaces and Gaussian policies
- Training stability and convergence challenges

**Key Concepts:** Policy optimization, gradient estimation, variance reduction, actor-critic

---

## 🎯 Learning Objectives

By working through these implementations, you will understand:

- ✅ How agents learn from rewards without explicit supervision
- ✅ The exploration-exploitation dilemma and resolution strategies
- ✅ Value-based vs. policy-based approaches and their trade-offs
- ✅ Temporal difference learning and its role in RL
- ✅ How to design reward functions and evaluate agent performance

## 🛠️ Technologies Used

- **Python 3.8+**
- **OpenAI Gym** - Standard RL environments
- **NumPy** - Numerical computations and Q-table storage
- **PyTorch / TensorFlow** - Neural network policies (policy gradients)
- **Matplotlib** - Reward curves and policy visualizations
- **Jupyter Notebooks** - Interactive experimentation

## 🚀 Getting Started
```bash
# Navigate to the reinforcement learning directory
cd 04-reinforcement-learning

# Install required packages
pip install gym numpy torch matplotlib jupyter

# Launch Jupyter Notebook
jupyter notebook
```

Then open any `.ipynb` file to explore the implementations.

## 📊 Repository Structure
```
04-reinforcement-learning/
├── policy-gradient-methods.ipynb
├── q-learning-implementation.ipynb
└── README.md (you are here)
```

## 💡 Key Insights

**The RL Framework:**
- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **State**: The current situation of the agent
- **Action**: Choices available to the agent
- **Reward**: Feedback signal indicating success/failure
- **Policy**: The agent's strategy (mapping states to actions)

**Q-Learning vs. Policy Gradients:**

| Aspect | Q-Learning | Policy Gradients |
|--------|------------|------------------|
| **What it learns** | Value function Q(s,a) | Policy π(a\|s) directly |
| **Action spaces** | Best for discrete | Handles continuous naturally |
| **Exploration** | Epsilon-greedy | Built into stochastic policy |
| **Convergence** | Strong guarantees (tabular) | Slower, local optima |
| **Scalability** | Limited (function approximation needed) | Scales with neural networks |

**Key Challenges in RL:**
- **Credit Assignment**: Which actions led to rewards?
- **Delayed Rewards**: Rewards may come long after critical decisions
- **Exploration**: Must try new actions to find better strategies
- **Sample Efficiency**: Learning from limited environment interactions

## 📚 Learning Resources

These implementations are inspired by and aligned with:
- **Stanford CS234: Reinforcement Learning** (Emma Brunskill)
- **Reinforcement Learning: An Introduction** (Sutton & Barto) - The RL bible
- **Deep Reinforcement Learning** (UC Berkeley CS285, Sergey Levine)
- **Spinning Up in Deep RL** (OpenAI)

## 🎓 Classic RL Environments Used

- **GridWorld**: Simple navigation tasks for Q-learning fundamentals
- **CartPole**: Classic control problem for policy gradients
- **MountainCar**: Continuous control with sparse rewards
- **Atari Games** (optional): Deep RL with visual inputs

## 🔗 Navigation

[← Previous: Natural Language Processing](../03-nlp/README.md) | [Next: Applied Projects →](../05-applied-projects/README.md)

---

**Part of the [AI Foundations Lab](../README.md) project - A self-directed journey through ML, DL, NLP, and RL.**

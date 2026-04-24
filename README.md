# 🚀 CartPole RL Sandbox

Welcome to the **CartPole RL Sandbox**! This is a simple, lightweight, and pure-JavaScript implementation of the classic Reinforcement Learning environment (CartPole), bundled with a custom-built Policy Gradient neural network—all running entirely in your browser without any external machine learning libraries! 🤯

![CartPole Dashboard Demo](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Cart-pole.svg/450px-Cart-pole.svg.png) <!-- Replace with an actual screenshot or GIF later! -->

## 🌟 Features

*   **🎮 Pure Vanilla JS**: No TensorFlow.js, no PyTorch, no external dependencies. The environment and the neural network are written from scratch!
*   **🎨 Stunning UI**: A sleek, dark-mode dashboard featuring glassmorphism, glowing neons, and modern typography.
*   **🧠 Live Neural Network Visualization**: Watch the weights of the hidden layer shift and change colors in real-time as the agent learns.
*   **📈 Performance Tracking**: A live, auto-updating chart showing the episode returns.
*   **🧪 Test Mode**: Switch off exploration at any time to see your agent perform pure inference.
*   **💾 Save & Load**: Found a hyperparameter combo that works? Save the trained model to a `.json` file and load it later!

## 🛠️ How it Works

1.  **The Environment**: The CartPole environment simulates a pole attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart.
2.  **The Goal**: Prevent the pole from falling over! A reward of `+1` is provided for every timestep that the pole remains upright.
3.  **The Brain**: A simple Feed-Forward Neural Network (1 hidden layer).
4.  **The Algorithm**: It uses **Policy Gradient (REINFORCE)**. The network outputs probabilities for moving left or right, and after every episode, it updates its weights via gradient ascent to increase the probability of actions that led to higher rewards.

## 🕹️ Usage

Simply open `cartpole_rl.html` in any modern web browser. 

### Controls

*   **Start / Pause**: Control the simulation loop.
*   **Test Mode Toggle**: When "Training Mode" is on, the agent samples actions based on probabilities (exploration). When "Testing Mode" is on, it greedily picks the best action (inference only).
*   **Simulation Speed**: Speed up the rendering to train faster!
*   **Hyperparameters**: Tweak the **Learning Rate**, **Gamma** (discount factor for future rewards), and the number of **Hidden Nodes**. Click **Apply Params / Reset** to restart the environment with your new settings.

---

## 🚀 Proposing Future Upgrades

Looking to take this sandbox to the next level? Here are some awesome ways to build upon it:

### 1. Upgrade the Algorithm (DQN / PPO)
Currently, the sandbox uses a basic Policy Gradient. You could implement **Deep Q-Learning (DQN)** with an Experience Replay Buffer, or even **Proximal Policy Optimization (PPO)** for much faster and more stable convergence.

### 2. Hardware Acceleration with TensorFlow.js
While building the network from scratch is educational, swapping the custom `PolicyNetwork` class out for **TensorFlow.js** would allow you to build deeper networks, utilize GPU acceleration, and solve much harder problems.

### 3. Add More Environments! 🌍
The code structure is already decoupled into an `env` and `policy`. You could easily add more classic control problems from the OpenAI Gym suite, such as:
*   🏔️ **MountainCar**
*   ⏱️ **Pendulum**
*   🌖 **LunarLander**

### 4. Interactive Perturbations 🌪️
Add mouse-drag events to the canvas! Allow the user to "flick" the pole or drag the cart while the agent is balancing it. This would perfectly demonstrate how robust the learned policy is against sudden, unexpected forces.

### 5. Multi-Agent Training / Genetic Algorithms 🧬
Instead of gradient descent, visualize 50 carts at once, all trying to balance the pole. Use a Genetic Algorithm (Neuroevolution) to select the best performing carts, mutate their weights, and breed the next generation!

---

*Built with ❤️ and Vanilla JavaScript.*

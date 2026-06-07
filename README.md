<div align="center">

![CartPole RL Sandbox](assets/hero.svg)

# 🚀 CartPole RL Sandbox

**A from-scratch reinforcement-learning playground that runs entirely in your browser — zero ML dependencies.**

![JavaScript](https://img.shields.io/badge/JavaScript-vanilla-f7df1e?logo=javascript&logoColor=000)
![Dependencies](https://img.shields.io/badge/dependencies-0-00f5d4)
![Algorithms](https://img.shields.io/badge/algorithms-REINFORCE%20%2B%20GA-7b61ff)
![Build](https://img.shields.io/badge/build%20step-none-ff6b9d)

</div>

---

## 🗺️ At a Glance

| | |
|---|---|
| 🎮 **Environment** | Classic CartPole physics — balance a pole on a moving cart |
| 🧠 **Brain** | Hand-rolled neural net: `4 → N → 2`, tanh + softmax, manual backprop |
| 🔀 **Two algorithms** | **REINFORCE** (policy gradient) and **Genetic Algorithm** (neuroevolution) |
| 🎨 **UI** | Dark glassmorphism dashboard with 4 live visualizations |
| 📦 **Stack** | 3 files · pure vanilla JS · no npm, no bundler, no TensorFlow |
| ▶️ **Run** | Open `cartpole_rl.html` in a browser — that's it |

---

## 🧩 Architecture

```mermaid
flowchart LR
    subgraph UI["🎨 cartpole_rl.html + style.css"]
        C[Controls & Hyperparams]
        V1[Canvas]
        V2[Return Chart]
        V3[Network SVG]
        V4[Metrics Panel]
    end

    subgraph CORE["🧠 script.js"]
        L["stepLoop (rAF)"]
        E[CartPole<br/>physics env]
        P[PolicyNetwork<br/>4 → N → 2]
        POP[Population<br/>N envs + nets]
    end

    C -->|live params| L
    L -->|REINFORCE| E
    L -->|GA| POP
    POP --> E
    E -->|state| P
    P -->|action probs| E
    L --> V1 & V2 & V3 & V4
```

---

## 🔁 How Learning Works

```mermaid
flowchart TD
    A([Start episode]) --> B[Observe state<br/>x, ẋ, θ, θ̇]
    B --> C{Mode?}
    C -->|Train| D[Sample action<br/>from softmax]
    C -->|Test| E[Argmax<br/>greedy action]
    D --> F[Step physics]
    E --> F
    F --> G{Pole fell or<br/>500 steps?}
    G -->|No| B
    G -->|Yes — REINFORCE| H[Discounted returns<br/>→ manual gradient ascent]
    G -->|Yes — GA| I[Rank fitness<br/>→ evolve next generation]
    H --> A
    I --> A
```

The agent earns **+1 per timestep** the pole stays upright. An episode ends when the cart drifts past ±2.4 units, the pole tips past 12°, or 500 steps elapse.

---

## 🌟 Features

- **🎮 Pure Vanilla JS** — the environment, network, *and* gradients are written by hand. No TensorFlow.js, no ONNX.
- **🧠 Live Network Visualization** — watch hidden-layer weights shift and recolor in real time as the agent learns.
- **🎯 Action Confidence** — the policy's left/right probabilities update continuously from the current prediction.
- **📈 Performance Tracking** — an auto-updating chart of episode returns (or best fitness in GA mode).
- **🧬 Two Training Modes** — switch between gradient-based REINFORCE and population-based neuroevolution at runtime.
- **🧪 Test Mode** — flip off exploration to watch pure greedy inference (REINFORCE only).
- **💾 Save & Load** — export trained weights to a `.json` file and reload them later. A pre-trained `cartpole_model.json` is included.

---

## 🕹️ Usage

Open `cartpole_rl.html` in any modern browser. No install step.

| Control | What it does |
|---|---|
| **Start / Pause** | Run or halt the simulation loop |
| **Algorithm** | Choose REINFORCE or Genetic Algorithm |
| **Test Mode** | Toggle exploration off (greedy inference) — REINFORCE only |
| **Simulation Speed** | Steps per frame — crank it up to train faster |
| **Learning Rate / Gamma / Hidden Nodes** | Live REINFORCE hyperparameters |
| **Population / Mutation σ** | Live GA hyperparameters |
| **Apply Params / Reset** | Restart the environment with new settings |

> 💡 All hyperparameters are **live** — they're read each frame (or each generation), so you can tune mid-training.

---

## 🧬 The Genetic Algorithm hook

`evolve(networks, fitnesses, popSize, sigma)` in `script.js` is an intentionally open **TODO**. The default placeholder applies mutation with no selection pressure, so it doesn't learn — wiring in selection, elitism, and crossover is the exercise. Helpers `mutate(net, sigma)` and `cloneNetwork(net)` are provided.

---

## 🚀 Future Upgrades

1. **Stronger algorithms** — DQN with experience replay, or PPO for faster, more stable convergence.
2. **More environments** — MountainCar 🏔️, Pendulum ⏱️, LunarLander 🌖 (the `env`/`policy` split already supports this).
3. **Interactive perturbations** — drag or flick the pole mid-balance to stress-test the learned policy.
4. **Finish the GA** — implement real selection and crossover in `evolve(...)`.

---

<div align="center">

*Built with ❤️ and Vanilla JavaScript.*

</div>

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

No build step or package manager needed. Open `cartpole_rl.html` directly in a browser. There are no npm packages, no bundlers, and no external ML dependencies — everything is pure vanilla JavaScript.

## Architecture

This is a single-file-per-concern web app split across three files: `cartpole_rl.html` (shell/layout), `script.js` (all logic, ~900 lines), and `style.css` (glassmorphism UI). The entire RL pipeline lives in `script.js`.

Two training algorithms are supported and selected at runtime via the **Algorithm** dropdown:
- **REINFORCE** — single-agent policy gradient (original).
- **Genetic Algorithm** — population-based neuroevolution (gradient-free).

### Core Classes in `script.js`

**`CartPole` (line 2)** — Physics environment.
- State vector: `[x, x_dot, theta, theta_dot]`
- Actions: 0 = left, 1 = right
- Terminates when cart moves > 2.4 units or pole angle > 12° from vertical, or after 500 steps

**`PolicyNetwork` (line 59)** — Hand-rolled single-hidden-layer neural network.
- Architecture: 4 → N (configurable, default 8) → 2
- Activations: tanh (hidden), softmax (output)
- Training: REINFORCE (policy gradient) with baseline normalization — no autograd library, all gradients computed manually
- In GA mode the network is used purely for inference; `trainEpisode` is never called.

**`Population` (line 216)** — N parallel `CartPole`+`PolicyNetwork` pairs for GA mode.
- `stepAll()` advances every still-alive cart one physics step.
- `allDone()` returns true once every cart has terminated (generation boundary).
- `nextGeneration(sigma)` calls `evolve(...)` to produce the next generation and resets all envs.
- `bestIndex()` returns the index of the cart with the highest fitness this generation. The global `policy` is repointed to this network each frame so the existing weight/probability visualizations track the leader automatically.

### Main Loop (`stepLoop`, line 845)

Driven by `requestAnimationFrame`. Each frame branches on `algorithm`:
- **REINFORCE**: steps env N times, on episode end computes discounted returns and calls `trainEpisode()`.
- **GA** (`stepLoopGA`, line 822): calls `population.stepAll()` N times; when `allDone()`, records best fitness and triggers `nextGeneration(sigma)`.

Both branches update all four live visualizations (canvas, chart, SVG network, metrics panel). In GA mode the canvas uses `drawEnvMulti` (line 560), which renders every alive cart semi-transparently and highlights the leader in gold.

### REINFORCE Training (`trainEpisode`, line 97)

Manual backprop through the policy gradient objective:
- Output delta for the taken action: `delta2[a] = (p[a] - 1) * G_t`
- Hidden layer delta via tanh derivative: `1 - h²`
- Weights updated directly: `W += lr * dW`

No optimizer state (no momentum, no Adam) — plain gradient ascent.

### GA Evolution (`evolve`, line 208)

Marked as a **user-editable TODO**. Signature: `evolve(networks, fitnesses, popSize, sigma) → PolicyNetwork[]`. Called once per generation; must return `popSize` fresh networks. Helpers available: `mutate(net, sigma)` (Gaussian-noise copy) and `cloneNetwork(net)` (exact copy). Selection, elitism, and crossover strategy are intentionally left open. The default placeholder is `networks.map(n => mutate(n, sigma))` — no selection pressure, so it does not learn.

### Action Selection (`chooseAction`, line 462)

REINFORCE mode only. GA mode uses greedy argmax inline in `Population.stepAll`.
- Training mode: stochastic sampling from softmax distribution
- Test mode: argmax (greedy) — toggled via the "Test Mode" button in the UI (hidden when GA is selected)

### Model Persistence

Save/load uses `JSON.stringify` on the weight matrices (`W1`, `b1`, `W2`, `b2`). A pre-trained model is checked into `cartpole_model.json`. In GA mode, **Save** writes the current best network in the population (since the global `policy` is repointed to the leader each frame).

## Key Constraints

- **No external ML libraries** — do not add TensorFlow.js, ONNX, or similar. The educational value is the from-scratch implementation.
- **Hyperparameters are live** — REINFORCE controls (learning rate, gamma, hidden nodes) and GA controls (population size, mutation σ) can be changed mid-training via the UI; the relevant fields are read each frame or generation.
- Algorithm-specific UI controls are toggled via `.reinforce-only` / `.ga-only` CSS classes by `applyAlgorithmVisibility()` (line 322). Metric box labels are also relabeled there (Episode → Generation, Return → Best Fit, Avg(50) → Mean Fit).
- All visualization state (chart history, SVG nodes/edges) is managed imperatively in global scope — there is no reactive framework or state management layer.
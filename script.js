// CartPole environment implementation
class CartPole {
    constructor() {
        this.gravity = 9.8;
        this.masscart = 1.0;
        this.masspole = 0.1;
        this.total_mass = this.masspole + this.masscart;
        this.length = 0.5;
        this.polemass_length = this.masspole * this.length;
        this.force_mag = 10.0;
        this.tau = 0.02;
        this.x_threshold = 2.4;
        this.theta_threshold_radians = 12 * Math.PI / 180;
        this.reset();
    }
    reset() {
        this.state = [
            (Math.random() * 2 - 1) * 0.05,
            (Math.random() * 2 - 1) * 0.05,
            (Math.random() * 2 - 1) * 0.05,
            (Math.random() * 2 - 1) * 0.05
        ];
        this.steps_beyond_done = null;
        return this.state.slice();
    }
    step(action) {
        let x = this.state[0];
        let x_dot = this.state[1];
        let theta = this.state[2];
        let theta_dot = this.state[3];

        let force = action === 1 ? this.force_mag : -this.force_mag;
        let costheta = Math.cos(theta);
        let sintheta = Math.sin(theta);

        let temp = (force + this.polemass_length * theta_dot * theta_dot * sintheta) / this.total_mass;
        let thetaacc = (this.gravity * sintheta - costheta * temp) /
                       (this.length * (4.0 / 3.0 - this.masspole * costheta * costheta / this.total_mass));
        let xacc  = temp - this.polemass_length * thetaacc * costheta / this.total_mass;

        x      = x + this.tau * x_dot;
        x_dot  = x_dot + this.tau * xacc;
        theta  = theta + this.tau * theta_dot;
        theta_dot = theta_dot + this.tau * thetaacc;

        this.state = [x, x_dot, theta, theta_dot];

        let done = false;
        if (x < -this.x_threshold || x > this.x_threshold ||
            theta < -this.theta_threshold_radians || theta > this.theta_threshold_radians) {
            done = true;
        }
        let reward = done ? 0 : 1;
        return { state: this.state.slice(), reward: reward, done: done };
    }
}

// Neural network for policy gradient
class PolicyNetwork {
    constructor(inputSize, hiddenSize, outputSize, learningRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.W1 = new Array(this.hiddenSize).fill(0).map(() =>
            new Array(this.inputSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.1));
        this.b1 = new Array(this.hiddenSize).fill(0);
        this.W2 = new Array(this.outputSize).fill(0).map(() =>
            new Array(this.hiddenSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.1));
        this.b2 = new Array(this.outputSize).fill(0);
    }

    forward(x) {
        const h = new Array(this.hiddenSize);
        for (let i = 0; i < this.hiddenSize; i++) {
            let sum = this.b1[i];
            for (let j = 0; j < this.inputSize; j++) {
                sum += this.W1[i][j] * x[j];
            }
            h[i] = Math.tanh(sum);
        }
        const z = new Array(this.outputSize);
        for (let k = 0; k < this.outputSize; k++) {
            let sum = this.b2[k];
            for (let i = 0; i < this.hiddenSize; i++) {
                sum += this.W2[k][i] * h[i];
            }
            z[k] = sum;
        }
        const maxZ = Math.max(...z);
        const expZ = z.map(v => Math.exp(v - maxZ));
        const sumExp = expZ.reduce((a, b) => a + b, 0);
        const probs = expZ.map(v => v / sumExp);
        return { probs: probs, h: h, z: z };
    }

    trainEpisode(states, actions, returns) {
        let dW1 = new Array(this.hiddenSize).fill(0).map(() => new Array(this.inputSize).fill(0));
        let db1 = new Array(this.hiddenSize).fill(0);
        let dW2 = new Array(this.outputSize).fill(0).map(() => new Array(this.hiddenSize).fill(0));
        let db2 = new Array(this.outputSize).fill(0);

        for (let t = 0; t < states.length; t++) {
            const x = states[t];
            const action = actions[t];
            const Gt = returns[t];
            const { probs, h, z } = this.forward(x);
            
            const delta2 = new Array(this.outputSize);
            for (let k = 0; k < this.outputSize; k++) {
                delta2[k] = probs[k];
                if (k === action) delta2[k] -= 1;
                delta2[k] *= Gt;
            }
            
            for (let k = 0; k < this.outputSize; k++) {
                for (let i = 0; i < this.hiddenSize; i++) {
                    dW2[k][i] += delta2[k] * h[i];
                }
                db2[k] += delta2[k];
            }
            
            const delta1 = new Array(this.hiddenSize).fill(0);
            for (let i = 0; i < this.hiddenSize; i++) {
                let sum = 0;
                for (let k = 0; k < this.outputSize; k++) {
                    sum += this.W2[k][i] * delta2[k];
                }
                delta1[i] = sum * (1 - h[i] * h[i]);
            }
            
            for (let i = 0; i < this.hiddenSize; i++) {
                for (let j = 0; j < this.inputSize; j++) {
                    dW1[i][j] += delta1[i] * x[j];
                }
                db1[i] += delta1[i];
            }
        }
        
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.inputSize; j++) {
                this.W1[i][j] += this.learningRate * dW1[i][j];
            }
            this.b1[i] += this.learningRate * db1[i];
        }
        for (let k = 0; k < this.outputSize; k++) {
            for (let i = 0; i < this.hiddenSize; i++) {
                this.W2[k][i] += this.learningRate * dW2[k][i];
            }
            this.b2[k] += this.learningRate * db2[k];
        }
    }
}

// ===== Genetic Algorithm helpers =====

function cloneNetwork(net) {
    const copy = new PolicyNetwork(net.inputSize, net.hiddenSize, net.outputSize, net.learningRate);
    copy.W1 = net.W1.map(row => row.slice());
    copy.b1 = net.b1.slice();
    copy.W2 = net.W2.map(row => row.slice());
    copy.b2 = net.b2.slice();
    return copy;
}

function gaussian() {
    const u = 1 - Math.random();
    const v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function mutate(net, sigma) {
    const m = cloneNetwork(net);
    for (let i = 0; i < m.hiddenSize; i++) {
        for (let j = 0; j < m.inputSize; j++) m.W1[i][j] += gaussian() * sigma;
        m.b1[i] += gaussian() * sigma;
    }
    for (let k = 0; k < m.outputSize; k++) {
        for (let i = 0; i < m.hiddenSize; i++) m.W2[k][i] += gaussian() * sigma;
        m.b2[k] += gaussian() * sigma;
    }
    return m;
}

// ===== USER TODO: implement evolution =====
//
// Called at the end of each generation, once every cart has terminated.
// `networks`  : array of PolicyNetwork (the current generation)
// `fitnesses` : array of numbers (total reward each cart earned, same order)
// `popSize`   : how many networks to return for the next generation
// `sigma`     : mutation magnitude (read from the UI)
//
// Must return a fresh array of `popSize` PolicyNetwork instances.
//
// Helpers you can use:
//   mutate(net, sigma)     -> returns a noise-perturbed COPY
//   cloneNetwork(net)      -> returns an exact COPY (no mutation)
//
// Design decisions worth thinking through:
//   1. SELECTION  - who gets to breed?
//        * Top-k:          sort by fitness, take best K.        Simple, greedy, can lose diversity.
//        * Tournament:     pick N at random, keep the best.     Tunable pressure via N.
//        * Roulette:       probability ∝ fitness.               Smooth, but weak when fitnesses are close.
//   2. ELITISM    - do the top performers survive UNCHANGED?
//        Without elitism the best can be lost to bad mutations. With too much, diversity collapses.
//   3. MUTATION   - mutate(parent, sigma) is provided.
//        Bigger sigma = more exploration but more chaos. The slider lets the user tune it.
function evolve(networks, fitnesses, popSize, sigma) {
    // TODO: write 5-15 lines implementing your chosen GA strategy.
    // Placeholder (random restart - terrible, will never learn) so the loop doesn't crash:
    return networks.map(n => mutate(n, sigma));
}

// ===== Population: runs N carts in lockstep =====

class Population {
    constructor(size, hiddenSize) {
        this.size = size;
        this.networks = [];
        this.envs = [];
        this.alive = [];
        this.fitnesses = [];
        this.steps = [];
        for (let i = 0; i < size; i++) {
            this.networks.push(new PolicyNetwork(4, hiddenSize, 2, 0));
            this.envs.push(new CartPole());
            this.alive.push(true);
            this.fitnesses.push(0);
            this.steps.push(0);
        }
        this.generation = 0;
        this.bestFitnessEver = 0;
    }

    stepAll() {
        for (let i = 0; i < this.size; i++) {
            if (!this.alive[i]) continue;
            const { probs } = this.networks[i].forward(this.envs[i].state);
            const action = probs[0] >= probs[1] ? 0 : 1;
            const { reward, done } = this.envs[i].step(action);
            this.fitnesses[i] += reward;
            this.steps[i] += 1;
            if (done || this.steps[i] >= 500) this.alive[i] = false;
        }
    }

    allDone() {
        return this.alive.every(a => !a);
    }

    bestIndex() {
        let best = 0;
        for (let i = 1; i < this.size; i++) {
            if (this.fitnesses[i] > this.fitnesses[best]) best = i;
        }
        return best;
    }

    nextGeneration(sigma) {
        const best = this.bestIndex();
        this.bestFitnessEver = Math.max(this.bestFitnessEver, this.fitnesses[best]);
        const newNetworks = evolve(this.networks, this.fitnesses, this.size, sigma);
        this.networks = newNetworks;
        for (let i = 0; i < this.size; i++) {
            this.envs[i].reset();
            this.alive[i] = true;
            this.fitnesses[i] = 0;
            this.steps[i] = 0;
        }
        this.generation += 1;
    }
}

// Global variables
const canvas = document.getElementById('cartpoleCanvas');
const ctx = canvas.getContext('2d');
const chartCanvas = document.getElementById('chartCanvas');
const chartCtx = chartCanvas.getContext('2d');
const networkSvg = document.getElementById('networkSvg');
const statusBadge = document.getElementById('statusBadge');
const probLeftFill = document.getElementById('probLeftFill');
const probRightFill = document.getElementById('probRightFill');
const probLeftVal = document.getElementById('probLeftVal');
const probRightVal = document.getElementById('probRightVal');
const policyConfidence = document.getElementById('policyConfidence');

// Inputs
const speedSlider = document.getElementById('speedSlider');
const speedVal = document.getElementById('speedVal');
const lrInput = document.getElementById('lrInput');
const gammaInput = document.getElementById('gammaInput');
const hiddenInput = document.getElementById('hiddenInput');
const testModeToggle = document.getElementById('testModeToggle');
const algoSelect = document.getElementById('algoSelect');
const popSizeInput = document.getElementById('popSizeInput');
const mutationSigmaInput = document.getElementById('mutationSigmaInput');
const testModeContainer = document.getElementById('testModeContainer');

speedSlider.oninput = () => {
    speedVal.textContent = speedSlider.value + 'x';
};

let env = new CartPole();
let learningRate = parseFloat(lrInput.value);
let gamma = parseFloat(gammaInput.value);
let hiddenSize = parseInt(hiddenInput.value);

let policy = new PolicyNetwork(4, hiddenSize, 2, learningRate);

let algorithm = algoSelect.value;
let population = null;

let episode = 0;
let episodeSteps = 0;
let episodeRewards = 0;
let episodeHistory = [];
let runningReturns = [];
let isPaused = true;
let isTestMode = false;
let latestActionProbs = [0.5, 0.5];

function applyAlgorithmVisibility() {
    const isGA = algorithm === 'ga';
    document.querySelectorAll('.reinforce-only').forEach(el => el.style.display = isGA ? 'none' : '');
    document.querySelectorAll('.ga-only').forEach(el => el.style.display = isGA ? '' : 'none');
    testModeContainer.style.display = isGA ? 'none' : '';
    document.querySelector('.metric-box:nth-child(1) .metric-label').textContent = isGA ? 'Generation' : 'Episode';
    document.querySelector('.metric-box:nth-child(3) .metric-label').textContent = isGA ? 'Best Fit' : 'Return';
    document.querySelector('.metric-box:nth-child(4) .metric-label').textContent = isGA ? 'Mean Fit' : 'Avg (50)';
}
applyAlgorithmVisibility();

algoSelect.addEventListener('change', () => {
    algorithm = algoSelect.value;
    applyAlgorithmVisibility();
    resetBtn.onclick();
});

// Buttons
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const saveBtn = document.getElementById('saveBtn');
const loadBtn = document.getElementById('loadBtn');
const fileInput = document.getElementById('fileInput');

startBtn.onclick = () => { isPaused = false; };
pauseBtn.onclick = () => { isPaused = true; };

testModeToggle.addEventListener('change', (e) => {
    isTestMode = e.target.checked;
    if (isTestMode) {
        statusBadge.textContent = "Testing Mode";
        statusBadge.classList.add('testing');
    } else {
        statusBadge.textContent = "Training Mode";
        statusBadge.classList.remove('testing');
    }
});

resetBtn.onclick = () => {
    learningRate = parseFloat(lrInput.value);
    gamma = parseFloat(gammaInput.value);
    hiddenSize = parseInt(hiddenInput.value);

    env = new CartPole();
    policy = new PolicyNetwork(4, hiddenSize, 2, learningRate);

    if (algorithm === 'ga') {
        const popSize = Math.max(2, parseInt(popSizeInput.value) || 25);
        population = new Population(popSize, hiddenSize);
        policy = population.networks[0];
    } else {
        population = null;
    }

    latestActionProbs = policy.forward(env.state).probs;

    episode = 0;
    episodeSteps = 0;
    episodeRewards = 0;
    episodeHistory = [];
    runningReturns = [];
    isPaused = true;
    initializeNetworkViz();
    updateMetrics();
    updatePolicyInsight();
    drawChart();
    updateNetworkViz();
    if (algorithm === 'ga') drawEnvMulti(); else drawEnv(env.state);
};

saveBtn.onclick = () => {
    const modelData = {
        hiddenSize: policy.hiddenSize,
        learningRate: policy.learningRate,
        W1: policy.W1,
        b1: policy.b1,
        W2: policy.W2,
        b2: policy.b2
    };
    const blob = new Blob([JSON.stringify(modelData)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cartpole_model.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

loadBtn.onclick = () => { fileInput.click(); };

fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        try {
            const data = JSON.parse(event.target.result);
            if (!data.W1 || !data.W2 || !data.hiddenSize) {
                alert('Invalid model file');
                return;
            }

            hiddenInput.value = data.hiddenSize;
            lrInput.value = data.learningRate || 0.01;
            
            policy = new PolicyNetwork(4, data.hiddenSize, 2, data.learningRate || 0.01);
            policy.W1 = data.W1;
            policy.b1 = data.b1;
            policy.W2 = data.W2;
            policy.b2 = data.b2;
            
            env = new CartPole();
            latestActionProbs = policy.forward(env.state).probs;
            episode = 0;
            episodeSteps = 0;
            episodeRewards = 0;
            episodeHistory = [];
            runningReturns = [];
            isPaused = true;
            
            initializeNetworkViz();
            updateNetworkViz();
            updateMetrics();
            updatePolicyInsight();
            drawChart();
            drawEnv(env.state);
            
            alert('Model loaded successfully!');
        } catch (err) {
            alert('Error loading model');
        }
    };
    reader.readAsText(file);
    fileInput.value = '';
};

function chooseAction(state) {
    const { probs } = policy.forward(state);
    latestActionProbs = probs.slice();
    
    if (isTestMode) {
        let maxProb = -1;
        let bestAction = 0;
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                bestAction = i;
            }
        }
        return bestAction;
    }

    const rnd = Math.random();
    let accum = 0;
    for (let i = 0; i < probs.length; i++) {
        accum += probs[i];
        if (rnd < accum) return i;
    }
    return probs.length - 1;
}

function computeReturns(rewards, gamma) {
    const returns = new Array(rewards.length);
    let G = 0;
    for (let t = rewards.length - 1; t >= 0; t--) {
        G = rewards[t] + gamma * G;
        returns[t] = G;
    }
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const std = Math.sqrt(returns.reduce((a, b) => a + (b - mean) * (b - mean), 0) / returns.length) || 1;
    return returns.map(r => (r - mean) / (std));
}

// Drawing functions
function drawEnv(state) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const worldWidth = env.x_threshold * 2;
    const scale = canvas.width / worldWidth;
    const cartY = canvas.height * 0.75;
    const cartWidth = 60;
    const cartHeight = 30;
    const poleLength = scale * (2 * env.length);

    const x = state[0];
    const theta = state[2];
    const cartX = (x + env.x_threshold) * scale - cartWidth / 2;

    // Draw track
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(20, cartY + cartHeight / 2);
    ctx.lineTo(canvas.width - 20, cartY + cartHeight / 2);
    ctx.stroke();

    // Draw cart
    ctx.fillStyle = '#38bdf8';
    ctx.shadowColor = '#38bdf8';
    ctx.shadowBlur = 15;
    ctx.beginPath();
    if (ctx.roundRect) {
        ctx.roundRect(cartX, cartY - cartHeight / 2, cartWidth, cartHeight, 6);
    } else {
        ctx.rect(cartX, cartY - cartHeight / 2, cartWidth, cartHeight); // Fallback
    }
    ctx.fill();
    ctx.shadowBlur = 0;

    // Draw pole
    const poleX = cartX + cartWidth / 2;
    const poleY = cartY - cartHeight / 2;
    const poleEndX = poleX + poleLength * Math.sin(theta);
    const poleEndY = poleY - poleLength * Math.cos(theta);
    
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.shadowColor = '#ef4444';
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.moveTo(poleX, poleY);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw axle
    ctx.fillStyle = '#f8fafc';
    ctx.beginPath();
    ctx.arc(poleX, poleY, 5, 0, 2 * Math.PI);
    ctx.fill();
}

function drawEnvMulti() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!population) return;

    const worldWidth = env.x_threshold * 2;
    const scale = canvas.width / worldWidth;
    const cartY = canvas.height * 0.75;
    const cartWidth = 30;
    const cartHeight = 16;
    const poleLength = scale * (2 * env.length);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(20, cartY + cartHeight / 2);
    ctx.lineTo(canvas.width - 20, cartY + cartHeight / 2);
    ctx.stroke();

    const bestIdx = population.bestIndex();
    for (let i = 0; i < population.size; i++) {
        const isBest = i === bestIdx;
        const isAlive = population.alive[i];
        if (!isAlive && !isBest) continue;

        const state = population.envs[i].state;
        const x = state[0];
        const theta = state[2];
        const cartX = (x + env.x_threshold) * scale - cartWidth / 2;
        const alpha = isBest ? 1.0 : (isAlive ? 0.25 : 0.0);

        ctx.fillStyle = isBest ? '#fbbf24' : `rgba(56, 189, 248, ${alpha})`;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(cartX, cartY - cartHeight / 2, cartWidth, cartHeight, 3);
        else ctx.rect(cartX, cartY - cartHeight / 2, cartWidth, cartHeight);
        ctx.fill();

        const poleX = cartX + cartWidth / 2;
        const poleY = cartY - cartHeight / 2;
        const poleEndX = poleX + poleLength * Math.sin(theta);
        const poleEndY = poleY - poleLength * Math.cos(theta);

        ctx.strokeStyle = isBest ? '#ef4444' : `rgba(239, 68, 68, ${alpha})`;
        ctx.lineWidth = isBest ? 6 : 3;
        ctx.beginPath();
        ctx.moveTo(poleX, poleY);
        ctx.lineTo(poleEndX, poleEndY);
        ctx.stroke();
    }
}

function drawChart() {
    chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
    
    // Draw axes
    chartCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    chartCtx.lineWidth = 1;
    chartCtx.beginPath();
    chartCtx.moveTo(40, 10);
    chartCtx.lineTo(40, chartCanvas.height - 25);
    chartCtx.lineTo(chartCanvas.width - 10, chartCanvas.height - 25);
    chartCtx.stroke();

    if (runningReturns.length === 0) return;

    const maxEpisodes = runningReturns.length;
    const maxVal = Math.max(...runningReturns, 10);
    const minVal = Math.min(...runningReturns, 0);

    const plotWidth = chartCanvas.width - 50;
    const plotHeight = chartCanvas.height - 35;

    // Create Gradient for fill
    const gradient = chartCtx.createLinearGradient(0, 0, 0, chartCanvas.height);
    gradient.addColorStop(0, 'rgba(16, 185, 129, 0.4)');
    gradient.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

    chartCtx.strokeStyle = '#10b981';
    chartCtx.lineWidth = 2;
    chartCtx.shadowColor = '#10b981';
    chartCtx.shadowBlur = 10;
    
    chartCtx.beginPath();
    chartCtx.moveTo(40, chartCanvas.height - 25);
    
    for (let i = 0; i < maxEpisodes; i++) {
        const x = 40 + (i / (maxEpisodes - 1 || 1)) * plotWidth;
        const yNorm = (runningReturns[i] - minVal) / (maxVal - minVal + 1e-6);
        const y = (chartCanvas.height - 25) - yNorm * plotHeight;
        if (i === 0) chartCtx.moveTo(x, y);
        else chartCtx.lineTo(x, y);
    }
    
    chartCtx.stroke();
    chartCtx.shadowBlur = 0;
    
    // Fill under the line
    chartCtx.lineTo(40 + plotWidth, chartCanvas.height - 25);
    chartCtx.lineTo(40, chartCanvas.height - 25);
    chartCtx.fillStyle = gradient;
    chartCtx.fill();

    // Draw labels
    chartCtx.fillStyle = '#94a3b8';
    chartCtx.font = '12px Inter';
    chartCtx.fillText('Episode', chartCanvas.width / 2 - 20, chartCanvas.height - 5);
    chartCtx.save();
    chartCtx.translate(15, chartCanvas.height / 2);
    chartCtx.rotate(-Math.PI / 2);
    chartCtx.fillText('Return', -20, 0);
    chartCtx.restore();
}

function initializeNetworkViz() {
    networkSvg.innerHTML = '';
    const width = parseInt(networkSvg.getAttribute('width'));
    const height = parseInt(networkSvg.getAttribute('height'));
    const marginX = 40;
    const layerX = [marginX, width / 2, width - marginX];

    const inputSize = policy.inputSize;
    const hiddenSize = policy.hiddenSize;
    const outputSize = policy.outputSize;

    const inputNodes = [];
    const hiddenNodes = [];
    const outputNodes = [];

    function layerPositions(n) {
        const spacing = height / (n + 1);
        const positions = [];
        for (let i = 0; i < n; i++) {
            positions.push((i + 1) * spacing);
        }
        return positions;
    }
    const inputYs = layerPositions(inputSize);
    const hiddenYs = layerPositions(hiddenSize);
    const outputYs = layerPositions(outputSize);

    for (let i = 0; i < inputSize; i++) {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', layerX[0]);
        circle.setAttribute('cy', inputYs[i]);
        circle.setAttribute('r', 8);
        circle.setAttribute('fill', '#38bdf8');
        networkSvg.appendChild(circle);
        inputNodes.push({ x: layerX[0], y: inputYs[i] });
    }
    for (let i = 0; i < hiddenSize; i++) {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', layerX[1]);
        circle.setAttribute('cy', hiddenYs[i]);
        circle.setAttribute('r', 10);
        circle.setAttribute('fill', '#f59e0b');
        networkSvg.appendChild(circle);
        hiddenNodes.push({ x: layerX[1], y: hiddenYs[i] });
    }
    for (let i = 0; i < outputSize; i++) {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', layerX[2]);
        circle.setAttribute('cy', outputYs[i]);
        circle.setAttribute('r', 12);
        circle.setAttribute('fill', '#10b981');
        networkSvg.appendChild(circle);
        outputNodes.push({ x: layerX[2], y: outputYs[i] });
    }

    networkSvg.weightLines1 = [];
    for (let i = 0; i < inputSize; i++) {
        for (let j = 0; j < hiddenSize; j++) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', inputNodes[i].x);
            line.setAttribute('y1', inputNodes[i].y);
            line.setAttribute('x2', hiddenNodes[j].x);
            line.setAttribute('y2', hiddenNodes[j].y);
            line.setAttribute('stroke', 'rgba(255,255,255,0.1)');
            line.setAttribute('stroke-width', 1);
            networkSvg.insertBefore(line, networkSvg.firstChild);
            networkSvg.weightLines1.push({ element: line, i: j, j: i });
        }
    }
    networkSvg.weightLines2 = [];
    for (let i = 0; i < hiddenSize; i++) {
        for (let j = 0; j < outputSize; j++) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', hiddenNodes[i].x);
            line.setAttribute('y1', hiddenNodes[i].y);
            line.setAttribute('x2', outputNodes[j].x);
            line.setAttribute('y2', outputNodes[j].y);
            line.setAttribute('stroke', 'rgba(255,255,255,0.1)');
            line.setAttribute('stroke-width', 1);
            networkSvg.insertBefore(line, networkSvg.firstChild);
            networkSvg.weightLines2.push({ element: line, k: j, i: i });
        }
    }
}

function updateNetworkViz() {
    let maxAbsW1 = 0;
    for (let i = 0; i < policy.hiddenSize; i++) {
        for (let j = 0; j < policy.inputSize; j++) {
            maxAbsW1 = Math.max(maxAbsW1, Math.abs(policy.W1[i][j]));
        }
    }
    let maxAbsW2 = 0;
    for (let k = 0; k < policy.outputSize; k++) {
        for (let i = 0; i < policy.hiddenSize; i++) {
            maxAbsW2 = Math.max(maxAbsW2, Math.abs(policy.W2[k][i]));
        }
    }
    const maxW = Math.max(maxAbsW1, maxAbsW2, 1e-6);
    
    networkSvg.weightLines1.forEach(item => {
        const weight = policy.W1[item.i][item.j];
        const norm = Math.abs(weight) / maxW;
        const thickness = 1 + 3 * norm;
        const color = weight >= 0 ? `rgba(16, 185, 129, ${norm})` : `rgba(239, 68, 68, ${norm})`;
        item.element.setAttribute('stroke', color);
        item.element.setAttribute('stroke-width', thickness);
    });
    
    networkSvg.weightLines2.forEach(item => {
        const weight = policy.W2[item.k][item.i];
        const norm = Math.abs(weight) / maxW;
        const thickness = 1 + 3 * norm;
        const color = weight >= 0 ? `rgba(16, 185, 129, ${norm})` : `rgba(239, 68, 68, ${norm})`;
        item.element.setAttribute('stroke', color);
        item.element.setAttribute('stroke-width', thickness);
    });
}

function updateMetrics() {
    if (algorithm === 'ga' && population) {
        document.getElementById('episode').textContent = population.generation;
        const aliveCount = population.alive.filter(a => a).length;
        document.getElementById('steps').textContent = aliveCount + '/' + population.size;
        document.getElementById('episodeReturn').textContent = population.bestFitnessEver.toFixed(0);
        const mean = population.fitnesses.reduce((a, b) => a + b, 0) / population.size;
        document.getElementById('avgReturn').textContent = mean.toFixed(1);
        return;
    }
    document.getElementById('episode').textContent = episode;
    document.getElementById('steps').textContent = episodeSteps;
    document.getElementById('episodeReturn').textContent = episodeRewards.toFixed(0);
    const last50 = runningReturns.slice(-50);
    const avg = last50.reduce((a, b) => a + b, 0) / (last50.length || 1);
    document.getElementById('avgReturn').textContent = avg.toFixed(2);
}

function updatePolicyInsight() {
    const left = latestActionProbs[0] || 0;
    const right = latestActionProbs[1] || 0;
    const confidence = Math.max(left, right);

    probLeftFill.style.width = `${(left * 100).toFixed(1)}%`;
    probRightFill.style.width = `${(right * 100).toFixed(1)}%`;
    probLeftVal.textContent = `${(left * 100).toFixed(1)}%`;
    probRightVal.textContent = `${(right * 100).toFixed(1)}%`;
    policyConfidence.textContent = `${(confidence * 100).toFixed(1)}%`;
}

function stepLoopGA() {
    const stepsPerFrame = parseInt(speedSlider.value);
    const sigma = parseFloat(mutationSigmaInput.value);

    for (let s = 0; s < stepsPerFrame; s++) {
        population.stepAll();
        if (population.allDone()) {
            const best = population.bestIndex();
            runningReturns.push(population.fitnesses[best]);
            population.nextGeneration(sigma);
        }
    }

    policy = population.networks[population.bestIndex()];
    latestActionProbs = policy.forward(population.envs[population.bestIndex()].state).probs;

    updateMetrics();
    updatePolicyInsight();
    drawChart();
    updateNetworkViz();
    drawEnvMulti();
    updateLiveExplainerState(population.envs[population.bestIndex()].state);
}

function stepLoop() {
    if (!isPaused) {
        if (algorithm === 'ga') {
            stepLoopGA();
            requestAnimationFrame(stepLoop);
            return;
        }
        const stepsPerFrame = parseInt(speedSlider.value);

        for (let s = 0; s < stepsPerFrame; s++) {
            const state = env.state;
            const action = chooseAction(state);
            const { state: nextState, reward, done } = env.step(action);

            episodeSteps += 1;
            episodeRewards += reward;

            if (!isTestMode) {
                episodeHistory.push({ state: state.slice(), action: action, reward: reward });
            }

            if (done || episodeSteps >= 500) {
                if (!isTestMode) {
                    const rewards = episodeHistory.map(item => item.reward);
                    const returns = computeReturns(rewards, gamma);
                    const states = episodeHistory.map(item => item.state);
                    const actions = episodeHistory.map(item => item.action);
                    policy.trainEpisode(states, actions, returns);
                    runningReturns.push(episodeRewards);
                }

                episode += 1;
                episodeSteps = 0;
                episodeRewards = 0;
                episodeHistory = [];
                env.reset();
            }
        }

        updateMetrics();
        updatePolicyInsight();
        drawChart();
        updateNetworkViz();
        drawEnv(env.state);
        updateLiveExplainerState(env.state);
    }
    requestAnimationFrame(stepLoop);
}

// Initialize
initializeNetworkViz();
updateNetworkViz();
latestActionProbs = policy.forward(env.state).probs;
updatePolicyInsight();
drawEnv(env.state);
drawChart();
updateMetrics();
requestAnimationFrame(stepLoop);

// Visual Explainer Modal & Real-Time Live Inspection
function updateLiveExplainerState(state) {
    const explainerModal = document.getElementById('explainerModal');
    if (!explainerModal || !explainerModal.classList.contains('active')) return;
    if (!state || state.length < 4) return;

    const x = state[0];
    const xDot = state[1];
    const theta = state[2];
    const thetaDot = state[3];

    const liveX = document.getElementById('liveStateX');
    const liveXDot = document.getElementById('liveStateXDot');
    const liveTheta = document.getElementById('liveStateTheta');
    const liveThetaDot = document.getElementById('liveStateThetaDot');

    if (liveX) liveX.textContent = x.toFixed(2);
    if (liveXDot) liveXDot.textContent = xDot.toFixed(2);
    if (liveTheta) liveTheta.textContent = (theta * 180 / Math.PI).toFixed(1) + '°';
    if (liveThetaDot) liveThetaDot.textContent = thetaDot.toFixed(2);

    const normX = Math.min(Math.max((x + 2.4) / 4.8, 0), 1) * 100;
    const barX = document.getElementById('barStateX');
    if (barX) { barX.style.width = `${normX}%`; }

    const normXDot = Math.min(Math.max((xDot + 3) / 6, 0), 1) * 100;
    const barXDot = document.getElementById('barStateXDot');
    if (barXDot) { barXDot.style.width = `${normXDot}%`; }

    const normTheta = Math.min(Math.max((theta + 0.21) / 0.42, 0), 1) * 100;
    const barTheta = document.getElementById('barStateTheta');
    if (barTheta) { barTheta.style.width = `${normTheta}%`; }

    const normThetaDot = Math.min(Math.max((thetaDot + 3) / 6, 0), 1) * 100;
    const barThetaDot = document.getElementById('barStateThetaDot');
    if (barThetaDot) { barThetaDot.style.width = `${normThetaDot}%`; }
}

const openExplainerBtn = document.getElementById('openExplainerBtn');
const sidebarExplainerBtn = document.getElementById('sidebarExplainerBtn');
const closeExplainerBtn = document.getElementById('closeExplainerBtn');
const explainerModalElement = document.getElementById('explainerModal');
const modalBtnClose = document.querySelector('.modal-btn-close');

function openExplainer() {
    if (explainerModalElement) {
        explainerModalElement.classList.add('active');
        const currentState = algorithm === 'ga' && population ? population.envs[population.bestIndex()].state : env.state;
        updateLiveExplainerState(currentState);
    }
}

function closeExplainer() {
    if (explainerModalElement) explainerModalElement.classList.remove('active');
}

if (openExplainerBtn) openExplainerBtn.addEventListener('click', openExplainer);
if (sidebarExplainerBtn) sidebarExplainerBtn.addEventListener('click', openExplainer);
if (closeExplainerBtn) closeExplainerBtn.addEventListener('click', closeExplainer);
if (modalBtnClose) modalBtnClose.addEventListener('click', closeExplainer);
if (explainerModalElement) {
    explainerModalElement.addEventListener('click', (e) => {
        if (e.target === explainerModalElement) closeExplainer();
    });
}
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeExplainer();
});

// Explainer Tabs
document.querySelectorAll('.explainer-tab').forEach(tabBtn => {
    tabBtn.addEventListener('click', () => {
        document.querySelectorAll('.explainer-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        
        tabBtn.classList.add('active');
        const targetId = `tab-${tabBtn.getAttribute('data-tab')}`;
        const targetPane = document.getElementById(targetId);
        if (targetPane) targetPane.classList.add('active');
    });
});

// Algo Pill Toggles
document.querySelectorAll('.algo-pill').forEach(pill => {
    pill.addEventListener('click', () => {
        document.querySelectorAll('.algo-pill').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.algo-explainer-content').forEach(c => c.style.display = 'none');
        
        pill.classList.add('active');
        const showId = pill.getAttribute('data-show');
        const targetContent = document.getElementById(showId);
        if (targetContent) targetContent.style.display = 'block';
    });
});

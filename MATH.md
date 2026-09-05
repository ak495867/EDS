# The Equilibrium Dissipation Signal (EDS) Model : Math & Implementation Guide

> **Who this is for:** you don't need a finance or math background to read this. Every symbol is defined before it's used, every formula gets a plain-English explanation right after it, and there's a worked numeric example near the end. If you already know quant finance, skim the notation table and jump to Section 4 onward.

---

## 0. The one-sentence idea

> Markets get "knocked" out of their resting state by news. **The knock itself — not the resting state — is where the tradeable signal lives.**

Think of a market like a **stretched rubber band**. News stretches it away from its natural resting length. The band doesn't snap back instantly — it pulls back over time, and *how fast and how hard it pulls back* tells you something about what's about to happen to the price. EDS is a model built entirely around measuring that stretch-and-snap-back process.

---

## 1. Notation glossary — read this before anything else

Every symbol used later is defined here first. Come back to this table any time you get lost.

| Symbol | Plain-English meaning | Type |
|---|---|---|
| $t$ | A point in time (e.g., a specific minute) | index |
| $X_t$ | Raw market data at time $t$ (price, volume, order book, etc.) | vector of numbers |
| $z_t^{eq}$ | The **equilibrium state** — the model's guess at what the market's "resting length" looks like right now | vector, dimension $d$ |
| $z_t^{obs}$ | The **observed state** — the model's guess at what the market's *actual current* state looks like, stretched or not | vector, dimension $d$ |
| $\Delta z_t$ | The **dissipation signal** — the gap/stretch between observed and equilibrium | vector, dimension $d$ |
| $\epsilon_t$, $\eta_t$ | Small random "noise" terms — things the model can't predict | random vectors |
| $V(\cdot)$ | A **potential function** — think "the energy stored in a stretched rubber band" | scalar function |
| $\lambda$ | A tuning number controlling how strongly the "band" pulls back to resting position | positive number |
| $f_\phi(X_t)$ | A small neural network that turns raw data into a "push" — how hard news is stretching the band right now | vector, dimension $d$ |
| $\hat{Y}_t$ | The model's final prediction (e.g., expected return) | scalar or vector |
| $g_\psi(\cdot)$, $\text{MLP}_\psi(\cdot)$ | Small neural networks (Multi-Layer Perceptrons) that turn internal signals into final predictions | function |
| $r_{t:t+h}$ | The market's actual return from time $t$ to $t+h$ | scalar |
| $\mathcal{L}$ | **Loss** — a number the model tries to make as small as possible during training | scalar |
| $\mathbb{R}^d$ | "A list of $d$ real numbers" — just means "a vector with $d$ numbers in it" | notation |

**A note on "vectors":** whenever you see something like $z_t^{eq} \in \mathbb{R}^d$, just read it as "$z_t^{eq}$ is a list of $d$ numbers." Neural networks work with lists of numbers (called *embeddings*) instead of raw prices, because lists of numbers can capture more nuanced patterns than a single price can.

---

## 2. The core hypothesis, formalized

**Hypothesis in words:** markets are usually near some slow-moving "normal" state. News pushes them away from that state. The most profitable moments are *while the market is snapping back*, not while it's calm.

### 2.1 The equilibrium state — the "resting length"

$$
z_t^{eq} \in \mathbb{R}^d
$$

**What this means:** at every moment $t$, the model keeps a running, slowly-updating estimate of "what does normal look like right now." It's like a very smooth, very slow-moving average — but computed on a rich internal representation of the market, not just on price.

### 2.2 The observed state — "what's actually happening"

$$
z_t^{obs} \in \mathbb{R}^d
$$

**What this means:** this is the model's read on the market's *actual* current condition — which may be far from "normal" right after big news.

### 2.3 The dissipation signal — "how far is the band stretched?"

$$
\Delta z_t = z_t^{obs} - z_t^{eq}
$$

**What this means:** simple subtraction. If observed and equilibrium are the same, $\Delta z_t = 0$ — no stretch, calm market. The bigger $\Delta z_t$ is, the more "stretched" the market currently is. This single quantity is the model's main novel ingredient.

### 2.4 How the equilibrium state evolves — "resting length barely changes"

$$
z_{t+1}^{eq} = z_t^{eq} + \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, \Sigma^{eq})
$$

**What this means:** the resting state only drifts a tiny, random amount each step ($\epsilon_t$ is small noise drawn from a normal/bell-curve distribution with covariance $\Sigma^{eq}$). This encodes the assumption "equilibrium changes slowly, not every minute."

### 2.5 How the observed state evolves — a "spring with a push"

This is the heart of the model. It's a **differential equation** — a formula for *how fast something is changing right now*, rather than its value directly.

$$
\frac{dz^{obs}}{dt} = \underbrace{-\lambda \nabla V(z_t^{obs} - z_t^{eq})}_{\text{restoring force}} + \underbrace{f_\phi(X_t)}_{\text{information impulse}} + \underbrace{\eta_t}_{\text{noise}}
$$

**Reading a differential equation for beginners:** $\dfrac{dz^{obs}}{dt}$ just means "the *rate of change* of $z^{obs}$ right now" — like how fast a car's position changes is its speed. This equation says: *the observed state's rate of change is the sum of three forces.*

- **Restoring force** $-\lambda \nabla V(\cdot)$: this is the rubber band pulling back toward equilibrium. $V$ is the "energy" stored in the stretch (a common, simple choice is $V(\Delta z) = \|\Delta z\|^2$, i.e., "energy grows with the square of the stretch," exactly like a real spring). $\nabla V$ (the *gradient*) just means "the direction that most reduces that energy" — pointing back toward equilibrium. $\lambda$ controls how stiff the spring is.
- **Information impulse** $f_\phi(X_t)$: a small neural network reads the raw market data and outputs "how hard is news pushing the market away from equilibrium right now."
- **Noise** $\eta_t$: unpredictable micro-level randomness (bid/ask jitter, small trades, etc.) that no model can capture.

### 2.6 What the model actually predicts

Two equivalent framings are offered:

**Framing A — predict the *rate of dissipation*:**

$$
\hat{Y}_t = g_\psi\!\left(\frac{d}{dt}\|\Delta z_t\|^2\right)
$$

**What this means:** $\|\Delta z_t\|^2$ is just "how big is the stretch" (squared length of the vector — the same idea as the Pythagorean theorem, generalized to $d$ dimensions). Taking its rate of change tells you whether the stretch is *growing* (market moving further from normal) or *shrinking* (snapping back). $g_\psi$ is a small neural network that converts that single number into a usable prediction.

**Framing B — predict the return directly from the stretch:**

$$
\hat{r}_{t:t+h} = \text{MLP}\!\left(\Delta z_t, \frac{d\Delta z_t}{dt}, \dots\right)
$$

**What this means:** feed the current stretch and how fast it's changing into a small neural network (an MLP is just "several layers of weighted sums + nonlinear squashing functions"), and get out a predicted return over the next $h$ time steps.

---

## 3. Where the novelty actually comes from

It helps to separate four different things a "new model" could mean — EDS only claims novelty in two of them.

| Question | Answer for EDS |
|---|---|
| **New neural network architecture?** | No. It reuses a standard tool (a Neural ODE — see box below) as plumbing, not as the contribution. |
| **New training objective (loss)?** | **Yes** — see Section 4. |
| **New inductive bias** (built-in assumption)? | **Yes, and this is the main contribution** — see below. |
| **New way of representing the market?** | **Yes** — the market is represented as a small dynamical system, not a snapshot of features. |

> **Box: what's a "Neural ODE"?** A normal neural network takes an input and produces an output in one step. A Neural ODE instead learns the *rate of change* (like Section 2.5) and then uses a numerical solver to "integrate forward in time" to see where the system ends up. It's the natural tool whenever you want to model something evolving continuously, like a spring, a temperature, or — here — a market state.

**The inductive bias, in one sentence:**
> *Markets have a slow-changing "normal" and fast, energy-releasing reactions to news — and the reaction dynamics themselves are the signal.*

This is different from giving a Transformer or an LSTM a pile of price/volume features and asking it to "figure out the pattern" — EDS instead *tells* the model the shape of the pattern (spring-like restoring force + external push) and only asks it to learn the details (how stiff is the spring, how big is the push).

**The new market representation** — instead of a flat feature vector, the market state is:

$$
M_t = \{ z_t^{\mathrm{obs}},\ z_t^{\mathrm{eq}},\ \dot{z}_t^{\mathrm{obs}},\ \Delta z_t \}
$$

**What this means:** rather than one snapshot, the model keeps track of *position* ($z^{obs}$), *reference point* ($z^{eq}$), *velocity* ($\dot z^{obs}$, i.e. rate of change), and *displacement* ($\Delta z_t$) — the same set of quantities you'd track for a physical object on a spring.

---

## 4. The loss function — how the model is actually trained

A **loss function** is just a number that measures "how wrong the model currently is." Training = adjusting the model's internal numbers (weights) to make this number smaller. EDS uses three losses added together:

$$
\mathcal{L} = \mathcal{L}_{\text{prediction}} + \lambda_1 \mathcal{L}_{\text{dynamics}} + \lambda_2 \mathcal{L}_{\text{stability}}
$$

**What this means:** the total loss is a weighted sum of three separate "complaints" the model has to satisfy at once. $\lambda_1$ and $\lambda_2$ are dials that control how much each complaint matters relative to the others.

### 4.1 Prediction loss — "did you predict the return correctly?"

Standard mean-squared error between predicted and actual returns. This is the only piece a normal forecasting model would have.

### 4.2 Dynamics loss — "did you obey the spring equation?"

$$
\mathcal{L}_{\text{dynamics}} = \left\| \frac{dz^{obs}}{dt} - \Big(-\lambda \nabla V(z_t^{obs}-z_t^{eq}) + f_\phi(X_t)\Big) \right\|^2
$$

**What this means:** this penalizes the model whenever its *actual* internal rate-of-change disagrees with what the spring-plus-push equation from Section 2.5 predicts it should be. This is called a **physics-informed loss** — it forces the internal machinery to actually behave like the physical story we told it to, rather than becoming an uninterpretable black box that happens to also predict returns.

### 4.3 Stability loss — "don't let 'normal' jump around"

$$
\mathcal{L}_{\text{stability}} = \left\| z_{t+1}^{eq} - z_t^{eq} \right\|^2
$$

**What this means:** penalizes big jumps in the equilibrium state from one step to the next, keeping it "slow-moving" as required by the hypothesis (Section 2.4).

---

## 5. The full pipeline, step by step

This is the order of operations a real implementation would follow, start to finish.

### Step 1 — Encode raw data

$$
X_t \xrightarrow{E_\theta} h_t, \qquad h_t \in \mathbb{R}^d
$$

**What this means:** raw numbers (price, volume, order-book snapshots for, say, the last 100 minutes) get passed through an encoder network $E_\theta$ that compresses them into a $d$-dimensional embedding $h_t$ — a denser, more useful internal representation.

### Step 2 — Infer the equilibrium state

$$
z_t^{eq} = \text{EMA}(h_t)
$$

**What this means:** EMA = **Exponential Moving Average**, a weighted average that gives more importance to recent values but never forgets the past entirely. Here it's applied to the embeddings $h_t$, not raw prices, so it can filter out short-term noise while adapting slowly. In practice this could be a small learnable network rather than a fixed formula.

### Step 3 — Evolve the observed state (the Neural ODE)

Start the observed state at the embedding: $z_{t_0}^{obs} = h_{t_0}$. Then integrate the spring equation forward:

$$
\frac{dz^{obs}}{dt} = -\lambda \nabla V(z^{obs}-z^{eq}) + f_\phi(X_t)
$$

A common simplification is a **quadratic potential**, which produces a **linear** restoring force — exactly the equation for a damped harmonic oscillator (a mass on a spring with friction), one of the most well-studied systems in physics:

$$
\nabla V(z^{obs}-z^{eq}) = z^{obs} - z^{eq}
$$

### Step 4 — Compute the dissipation signal after a short time step

Solve the equation above forward by a short horizon $\Delta t$ (e.g., 5 time steps) to get $z_{t+\Delta t}^{obs}$, then compute how fast the "stretch energy" is changing:

$$
\Delta E_t = \frac{\|z_{t+\Delta t}^{obs}-z_{t+\Delta t}^{eq}\|^2 - \|z_t^{obs}-z_t^{eq}\|^2}{\Delta t}
$$

**What this means:** this is literally "(energy now) minus (energy a moment ago), divided by the time elapsed" — the definition of a *rate*. If $\Delta E_t > 0$, the market is stretching further from normal (accelerating reaction). If $\Delta E_t < 0$, it's snapping back (dissipating).

### Step 5 — Make the final prediction

$$
\hat{Y}_t = \text{MLP}_\psi\left(\Delta E_t,\ z_t^{obs},\ z_t^{eq},\ \Delta z_t\right)
$$

which can output one or several targets at once:

$$
\begin{bmatrix} \mathbb{E}[r_{t:t+h}] \\ P(r_{t:t+h}>0) \\ \sigma_{t:t+h} \end{bmatrix}
$$

**What this means:** the model can simultaneously output (1) the *expected* return, (2) the *probability* the return is positive, and (3) the expected *volatility* (riskiness) — all from the same dissipation-based internal state.

### Step 6 — Turn the prediction into a trading position

A simple **Kelly Criterion** sizing rule converts the prediction into a portfolio weight (how much of your capital to allocate):

$$
w_t = \frac{\mathbb{E}[r_{t:t+h}]}{\sigma_{t:t+h}^2}
$$

**What this means:** bet more when expected return is high *relative to* risk, and less when risk is high. This weight is then passed through additional risk constraints before actual trades are placed — the model never sizes trades on raw conviction alone.

---

## 6. A tiny worked numeric example

To make Sections 2–5 concrete, here's a toy example using just **1-dimensional** states (i.e., $d=1$, a single number instead of a vector) — real models use $d$ in the dozens or hundreds, but the arithmetic is identical.

Suppose at time $t$:
- Equilibrium state: $z_t^{eq} = 10.0$
- Observed state: $z_t^{obs} = 10.6$
- So the stretch is: $\Delta z_t = z_t^{obs} - z_t^{eq} = 0.6$

Using the simple quadratic potential $V(\Delta z) = \Delta z^2$, the "stretch energy" right now is:

$$
\|\Delta z_t\|^2 = 0.6^2 = 0.36
$$

Say the restoring-force stiffness is $\lambda = 0.5$, and the news-impulse network currently outputs $f_\phi(X_t) = 0.1$ (a small extra push away from equilibrium). Then the rate of change of the observed state is:

$$
\frac{dz^{obs}}{dt} = -\lambda(z^{obs}-z^{eq}) + f_\phi(X_t) = -0.5(0.6) + 0.1 = -0.2
$$

**Reading this:** the observed state is currently moving *toward* equilibrium at a rate of $0.2$ per unit time (the negative sign means "shrinking the gap") — even though a small news push is still nudging it the other way. The restoring force is winning. If we take one small time step, say $\Delta t = 1$:

$$
z_{t+1}^{obs} \approx z_t^{obs} + \frac{dz^{obs}}{dt}\cdot \Delta t = 10.6 - 0.2 = 10.4
$$

New stretch: $\Delta z_{t+1} = 10.4 - 10.0 = 0.4$, so energy is now $0.4^2 = 0.16$. The rate of dissipation is:

$$
\Delta E_t = \frac{0.16 - 0.36}{1} = -0.20
$$

**Reading this:** energy dropped — the "band" is snapping back, which under the model's hypothesis is exactly the moment it expects to have predictive information about the next price move.

---

## 7. Evaluating whether it actually works (in plain language)

A physics-flavored story is not evidence by itself. The evaluation plan is designed to test it rigorously against normal baselines (Linear Regression, XGBoost, LSTM, Transformer — all fed the *same* raw data).

| Metric | Plain-English meaning |
|---|---|
| **Information Coefficient (IC)** | The Spearman rank correlation between the model's predicted returns and what actually happened. Roughly: "if I sort assets by predicted return, do they come out in roughly the same order as their actual return?" Ranges from $-1$ (backwards) to $1$ (perfect). |
| **IC decay** | How fast the IC weakens the further out you try to predict. A signal that's great for 1 minute but useless for 1 hour has fast IC decay. |
| **Sharpe Ratio** | Return earned per unit of risk taken, *after* subtracting trading costs. Higher is better; it's the standard way to compare strategies that take different amounts of risk. |
| **Regime sensitivity** | Does the model do especially well specifically during high-volatility, "stretched" periods? If the hypothesis is right, it should — this is the most direct test of the core idea. |
| **Diebold-Mariano test** | A statistical test asking: "is Model A actually more accurate than Model B, or could the difference just be luck?" |
| **Permutation test on Sharpe Ratio** | Shuffles the data many times and recomputes performance, to check whether the strategy's edge could have appeared by chance. |

**Validation method:** *Purged Walk-Forward Validation* — the model is only ever tested on data that comes chronologically *after* the data it was trained on (never mixing past and future), with a gap ("purge") around the split to prevent subtle leakage from overlapping windows.

---

## 8. Recap — why this counts as a real research question

The claim is **not** "a new neural network was invented." The claim is:

> *Markets behave like non-equilibrium dynamical systems — a slow-moving equilibrium plus fast, information-driven, energy-dissipating reactions — and explicitly modeling that structure (rather than treating market data as a flat feature vector) improves out-of-sample return prediction versus standard baselines.*

That's a falsifiable, testable claim, and the entire point of Sections 6 and 7 is that the evaluation plan can actually prove it *wrong* if the hypothesis is bad — which is what makes it science rather than a plausible-sounding story.

---

## 9. Mini glossary (quant + math terms used above)

| Term | Meaning |
|---|---|
| **Latent state** | An internal, learned representation that isn't directly observed in the raw data — the model invents it to make its job easier. |
| **Embedding** | A list of numbers a neural network uses to represent something (a word, an image patch, or here, a slice of market data). |
| **Gradient** ($\nabla$) | The direction of steepest increase of a function; $-\nabla V$ points toward the *decrease* — i.e., "downhill." |
| **Differential equation** | An equation describing a *rate of change* rather than a value directly; solving it tells you the value over time. |
| **Neural ODE** | A model where a neural network defines the rate-of-change function of a differential equation, solved with a numerical integrator. |
| **MLP (Multi-Layer Perceptron)** | The simplest kind of neural network: alternating weighted sums and nonlinear "squashing" functions. |
| **Exponential Moving Average (EMA)** | A running average that weights recent data more heavily but never fully forgets older data. |
| **Mean Squared Error (MSE)** | Average of the squared differences between predictions and actual values — a standard way to measure prediction error. |
| **Spearman rank correlation** | Correlation computed on the *rankings* of two variables rather than their raw values — robust to outliers and nonlinearity. |
| **Sharpe Ratio** | (Average return − risk-free rate) ÷ (standard deviation of returns) — return per unit of risk. |
| **Walk-forward validation** | Testing a model only on data chronologically after its training data, repeated across multiple rolling windows. |
| **Kelly Criterion** | A bet-sizing formula that scales position size with expected edge and inversely with risk (variance). |

---

## 10. Notes on scope

This document explains the **mathematical structure** of EDS. It intentionally does not include: hyperparameter values for a specific implementation, transaction cost modeling, execution/slippage assumptions, or live portfolio risk controls — those belong in an implementation-specific document once the research prototype above is validated.

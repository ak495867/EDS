# EDS — Equilibrium Dissipation Signal Model
### A Beginner-Friendly Mathematical & Implementation Guide

> **Who this is for:** you don't need a PhD in physics or a background in stochastic calculus to read this. Every symbol is defined the first time it appears, every formula is followed by a plain-English translation, and every section builds on the one before it. If you know what a "moving average" and a "for loop" are, you can follow this document end to end.

---

## 1. What EDS Is Trying to Do

Most trading models ask: *"given these numbers, what happens next?"* They throw price history, volume, and maybe some technical indicators into a model (Random Forest, LSTM, Transformer) and let it find patterns.

**EDS asks a different, more specific question**, based on a hypothesis about *why* prices move at all:

> **Hypothesis:** every asset has a slow-moving, hard-to-observe "fair value" — call it the **equilibrium state**. Left alone, the price would drift smoothly toward this fair value. But news, rumors, and shocks constantly "kick" the price away from equilibrium. The size and freshness of that kick — the **dissipation** — tells you how much correction (reversion, or continuation) is likely to happen next, and in which direction.

In plain terms: think of the equilibrium state like the resting level of water in a bathtub, and news events like someone splashing the water. EDS tries to learn (a) where the resting level is, and (b) how big the current splash is — because a big recent splash means a lot of movement is still "in motion" and hasn't settled yet.

This gives EDS two things a plain price-history model doesn't have:

1. A learned notion of **"fair value"** that isn't just a moving average — it's inferred from the data itself.
2. A **dissipation signal**, coming from news/text, that measures how far and how violently the price has been pushed away from that fair value.

The final trading signal is built from the *gap* between price and equilibrium, adjusted by how much "settling" energy is still left in the system.

---

## 2. The Cast of Characters (Notation Glossary)

Read this section once, then use it as a lookup table while reading the rest.

| Symbol | Plain-English meaning |
| --- | --- |
| $t$ | A point in time (e.g., a specific day or hour) |
| $x_t$ | The observed variable at time $t$ — usually price or return |
| $z_t$ | The **latent equilibrium state** at time $t$: EDS's internal, unobservable estimate of "fair value" |
| $e_t$ | The **deviation**: how far the observed price is from the equilibrium ($e_t = x_t - z_t$) |
| $n_t$ | Raw news/text data available at time $t$ (headlines, articles, filings) |
| $d_t$ | The **dissipation signal**: a number (or small vector) summarizing how much "shock energy" news has injected into the system recently |
| $f_\theta(\cdot)$ | A neural network (parameters $\theta$) that describes how the equilibrium state drifts on its own, with no news |
| $g_\phi(\cdot)$ | A neural network (parameters $\phi$) that turns raw news into the dissipation signal $d_t$ |
| $h_\psi(\cdot)$ | A neural network (parameters $\psi$) that turns news embeddings *and* the current state into a forcing push on the equilibrium |
| $D_t$ | **Dissipation energy**: a single non-negative number measuring the "size" of the current shock, $D_t = \lVert d_t \rVert^2$ |
| $\frac{dz}{dt}$ | "The rate of change of $z$ over time" — how fast the equilibrium state is drifting, at this instant |
| $\mathcal{N}(\mu, \sigma^2)$ | A normal (bell-curve) distribution with mean $\mu$ and variance $\sigma^2$ |
| $\mathbb{E}[\cdot]$ | "The expected value of" — roughly, the average outcome if you repeated the experiment many times |
| $\mathcal{L}$ | A **loss function**: a number the model tries to make as small as possible during training |
| $\hat{y}$ | A model's prediction (the "hat" means "estimated") |
| $\sigma(\cdot)$ | The sigmoid function, $\sigma(u) = \frac{1}{1+e^{-u}}$, which squashes any real number into the range $(0,1)$ — used to turn a raw score into a probability |

---

## 3. Step 1 — Turning Prices into Something a Model Can Learn From

### 3.1 Returns, not raw prices

Raw prices (e.g., "$187.42") are hard for models to learn from because their scale drifts over years. Instead we use **returns**:

$$
r_t = \frac{C_t}{C_{t-1}} - 1,
$$

where $C_t$ is the closing price on day $t$. This says: *"what fraction did the price change by, from yesterday to today?"* A return of $0.02$ means "up 2%."

### 3.2 The label we're trying to predict

EDS predicts the **direction** of the next return:

$$
y_{t+1} = \mathbf{1}\{ r_{t+1} > 0 \},
$$

where $\mathbf{1}\{\cdot\}$ is 1 if the condition in the braces is true, and 0 otherwise. In words: *"did the price go up tomorrow, yes or no?"*

### 3.3 News as data

Alongside price data, EDS consumes a stream of news/text $n_t$ (headlines, articles) tagged to the relevant asset at time $t$. This is converted into a numeric **embedding** — a fixed-length vector of numbers that captures the "meaning" of the text — using a standard pretrained text-embedding model. Call this embedding $v_t \in \mathbb{R}^k$. This step is a preprocessing step and is not itself novel; the novel part is what EDS *does* with $v_t$ next (Section 5).

---

## 4. Step 2 — The Core Idea: Equilibrium as a Differential Equation

### 4.1 What is a "differential equation," in plain terms?

A differential equation just describes **how fast something changes**, rather than what it equals directly. For example, "a car accelerates at 2 meters per second, per second" is a differential-equation-style statement — it tells you the *rate of change* of speed, not the speed itself. You then have to "integrate" (add up all those small changes over time) to find out where the car actually ends up.

EDS models the equilibrium state $z_t$ the same way — not by predicting its value directly, but by predicting how it *drifts*:

$$
\frac{dz_t}{dt} = f_\theta(z_t) + h_\psi(z_t, v_t).
$$

Let's unpack the right-hand side term by term:

- $f_\theta(z_t)$ — the **self-drift** term. Left undisturbed, this pulls $z_t$ toward a stable resting point, the same way a ball in a bowl rolls toward the bottom. $f_\theta$ is a small neural network that learns the shape of that "bowl" directly from data — nobody tells it in advance what the fair value should be.
- $h_\psi(z_t, v_t)$ — the **news-forcing** term. This is the "splash" — it pushes $z_t$ away from its resting point whenever news comes in that's surprising or significant. $h_\psi$ is a second small neural network that decides *how big and in what direction* the push should be, based on what the news embedding $v_t$ says and where the state currently is.

If there were never any news, $h_\psi$ would contribute (approximately) zero, and $z_t$ would just glide toward the bottom of its bowl. Because news arrives continuously, $z_t$ is always being pushed around a bit before it can fully settle — which is exactly the "non-equilibrium" behavior real markets show.

### 4.2 Solving the equation: the Neural ODE

We can't write $f_\theta$ and $h_\psi$ down as neat formulas — they're neural networks with thousands of parameters, learned from data. But we can still compute $z_t$ at any future time by **numerically integrating** the equation forward, one small time-step $\Delta$ at a time. This general technique is called a **Neural ODE** (Ordinary Differential Equation solved with a neural network inside it). The simplest way to do this (Euler's method) is:

$$
z_{t+\Delta} \approx z_t + \Delta \cdot \Big[ f_\theta(z_t) + h_\psi(z_t, v_t) \Big].
$$

In words: *"the equilibrium state a little later is roughly the state now, plus a small step in the direction the equation says it's moving."* Real implementations use more accurate solvers (e.g., Runge-Kutta / `dopri5`), but the intuition is identical — just with smaller, smarter steps.

### 4.3 Getting from price to the starting point $z_0$

Before we can integrate forward, we need a starting value $z_0$. This comes from an **encoder** network that looks at a recent window of prices and returns:

$$
z_0 = \mu_\phi(x_{t-L:t}), \qquad L = \text{window length (e.g., 20 days)}.
$$

In words: *"look at the last $L$ days of price data and produce a best guess for today's fair value."* This is analogous to how a moving average summarizes recent prices — except the encoder is a neural network that can learn a much richer summary than a simple average.

---

## 5. Step 3 — The Dissipation Signal: Turning News Into a Number

This is the heart of what makes EDS different from a generic Neural ODE forecaster.

### 5.1 From news embedding to "shock vector"

Recall $v_t$ is the numeric embedding of the news at time $t$. We pass it (together with the current state) through the forcing network:

$$
d_t = h_\psi(z_t, v_t) \in \mathbb{R}^m.
$$

$d_t$ is a small vector that represents *"the direction and strength of the push that news is currently applying to the equilibrium state."*

### 5.2 Dissipation energy — a single number you can actually look at

A vector is hard to interpret at a glance, so EDS also defines a scalar (single-number) summary called the **dissipation energy**:

$$
D_t = \lVert d_t \rVert^2 = \sum_{i=1}^{m} d_{t,i}^2 .
$$

This is just "add up the squares of every entry in the vector." The physical analogy: if $d_t$ is like a force, $D_t$ is like the energy that force is currently injecting into the system. A big $D_t$ means "a lot of shock is happening right now"; a small $D_t$ means "things are calm, prices are close to drifting on their own."

**Why does this matter for predicting returns?** The hypothesis is: *when $D_t$ is high, the system is "away from rest" and has to move (settle) at some point soon — so the near-future return is more predictable than during calm periods.* When $D_t$ is near zero, the market is close to its own equilibrium and near-term moves are closer to noise.

### 5.3 The deviation — how far price has strayed from fair value

We also track the plain gap between the observed price/return and the model's internal fair value:

$$
e_t = x_t - z_t .
$$

A positive $e_t$ means "the market price is trading above what EDS thinks is fair" (potentially overextended); a negative $e_t$ means the opposite.

---

## 6. Step 4 — From Signals to a Prediction (The Alpha Head)

EDS doesn't hand $z_t$, $e_t$, and $D_t$ directly to a trader — it combines them through one more small neural network, the **alpha head** $a_\omega$, which produces the final probability:

$$
p_t = \sigma\Big( a_\omega\big(z_t,\ e_t,\ D_t\big) \Big).
$$

Walking through this:

1. $a_\omega(z_t, e_t, D_t)$ takes the three signals and combines them into one raw score (a real number, could be positive or negative, no fixed range).
2. $\sigma(\cdot)$, the sigmoid function, squashes that raw score into a probability between 0 and 1.
3. $p_t$ is interpreted as *"the model's estimated probability that tomorrow's return is positive."*

A natural, interpretable special case (useful for intuition, and a reasonable first baseline before training the full $a_\omega$ network) is:

$$
p_t \approx \sigma\Big( -\beta_1 \, e_t \;+\; \beta_2 \, D_t \cdot \text{sign}(-e_t) \Big),
$$

which encodes two plain-language ideas at once:

- **Mean reversion:** if price is above fair value ($e_t > 0$), lean toward predicting a *down* move, and vice versa — this is the $-\beta_1 e_t$ term.
- **Dissipation-scaled conviction:** the higher the current dissipation energy $D_t$, the *more strongly* we lean on that reversion signal, because a large recent shock means more "unsettled" movement is still working through the system.

The actual $a_\omega$ used in the full model is a small multi-layer neural network rather than this fixed formula, so it can learn nonlinear combinations and interactions the simple formula above can't capture — but this expression is a good mental model for *why* the three inputs matter.

---

## 7. Step 5 — How the Whole Thing Is Trained

Training means: adjust all the neural network parameters ($\theta$, $\phi$, $\psi$, $\omega$) so that the model's outputs match reality as closely as possible, across many historical examples. EDS is trained on two goals simultaneously:

### 7.1 Goal 1 — Reconstruction (make sure $z_t$ is meaningful)

We don't want $z_t$ to be an arbitrary internal number that happens to help predictions by accident (that could just be memorizing noise). We want it to actually behave like a "fair value" for the price series. So we ask a decoder network $\text{dec}_\psi$ to reconstruct the actual observed price from $z_t$, and penalize it when it can't:

$$
\mathcal{L}_{\text{recon}} = \frac{1}{N}\sum_{t=1}^{N} \big( x_t - \text{dec}_\psi(z_t) \big)^2 .
$$

This is ordinary **mean squared error**: for each time step, take the difference between the real value and the model's reconstruction, square it (so positive and negative errors both count, and big errors are punished more), and average over all $N$ examples.

### 7.2 Goal 2 — Direction prediction (the actual trading task)

$$
\mathcal{L}_{\text{pred}} = -\frac{1}{N}\sum_{t=1}^{N} \Big[ y_{t+1}\log p_t + (1-y_{t+1})\log(1-p_t) \Big].
$$

This is **binary cross-entropy**, the standard loss for "predict a yes/no outcome." In plain terms: it heavily penalizes the model for being confidently wrong (e.g., predicting $p_t = 0.95$ when the answer was actually "down"), and rewards it for being confidently right.

### 7.3 Combining the two goals

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda \, \mathcal{L}_{\text{pred}},
$$

where $\lambda$ is a hand-set weight (a hyperparameter) controlling how much we prioritize prediction accuracy versus keeping $z_t$ faithful to the real price series. Training proceeds by standard gradient descent (e.g., the Adam optimizer): repeatedly nudge $\theta, \phi, \psi, \omega$ in the direction that shrinks $\mathcal{L}$, using many small batches of historical data.

**Important nuance:** because $z_t$ is produced by integrating an ODE forward through time, gradients flow back not just through one time step but through the *entire integration path* — this is what "Neural ODE" training means in practice, and it's typically done efficiently with the *adjoint method*, which avoids having to store every intermediate integration step in memory.

---

## 8. Putting It All Together: The Full Pipeline

$$
\text{price history} \;\rightarrow\; \text{encoder} \;\rightarrow\; z_0
$$
$$
\text{news embeddings} \;\rightarrow\; h_\psi \;\rightarrow\; d_t,\ D_t
$$
$$
z_0,\ f_\theta,\ h_\psi \;\xrightarrow{\text{Neural ODE integration}}\; z_t \text{ for each future } t
$$
$$
(z_t,\ e_t,\ D_t) \;\rightarrow\; a_\omega \;\rightarrow\; p_t \;\rightarrow\; \text{trade / no-trade decision}
$$

| Stage | What goes in | What comes out | Plain-English role |
| --- | --- | --- | --- |
| Encoder | recent price window | $z_0$ | "Best guess at today's fair value" |
| News forcing network ($h_\psi$) | news embedding + state | $d_t$, $D_t$ | "How big is the current shock, and which way is it pushing?" |
| Self-drift network ($f_\theta$) | current state | drift direction | "Where would price settle if left alone?" |
| Neural ODE integrator | $z_0$, $f_\theta$, $h_\psi$ | $z_t$ for future $t$ | "Simulate the equilibrium state forward in time" |
| Decoder | $z_t$ | reconstructed price | "Sanity check: does $z_t$ still look like a real price?" |
| Alpha head ($a_\omega$) | $z_t$, $e_t$, $D_t$ | $p_t$ | "Turn everything into a single up/down probability" |

---

## 9. Turning a Probability Into a Trading Decision

As in most probability-based systems, EDS avoids forcing a decision when it isn't confident. Using a confidence threshold $\tau$ (e.g., $\tau = 0.6$):

$$
\hat{y} =
\begin{cases}
\text{long (predict up)}, & p_t \ge \tau \\
\text{short (predict down)}, & p_t \le 1-\tau \\
\text{abstain (no trade)}, & 1-\tau < p_t < \tau
\end{cases}
$$

In words: *"only take a position when the model is meaningfully more confident than a coin flip; otherwise, sit out."* This trades off **coverage** (how often you trade) against **selective accuracy** (how often you're right, on the trades you do make) — a model that only trades on its 5 most confident days a year might have very high accuracy but very little practical use, so both numbers should always be reported together, not just one.

---

## 10. Why This Is a *Novel* Hypothesis, Not Just Another Architecture

- **The inductive bias comes from a stated, falsifiable belief about markets** — that price behavior is best understood as a slow equilibrium process periodically kicked off-course by external information — rather than "let's see what a big enough neural net finds."
- **The equilibrium state $z_t$ is learned, not assumed.** It isn't a moving average or a hand-picked "fair value" formula; the self-drift network $f_\theta$ discovers the shape of the "bowl" from the data itself.
- **News enters the model physically, not just as another feature column.** Instead of concatenating a sentiment score onto a feature vector, news is treated as a *forcing term* that perturbs a dynamical system — a fundamentally different mechanism than typical feature-based sentiment models.
- **Dissipation energy $D_t$ is a genuinely new, interpretable quantity.** It isn't simply "how strong was the news" (a text-based measure) or "how much did price move" (an already-realized outcome) — it's the model's own internal estimate of how much unresolved disequilibrium currently exists, computed *before* the resulting price move has fully played out.
- **The model is inspectable.** You can plot $z_t$ against the real price to sanity-check the learned "fair value," plot $D_t$ over time to see when the model thinks the market is most unsettled, and directly examine cases where large $D_t$ did or didn't precede a big move.

---

## 11. Honest Limitations

- **Identifiability:** in principle, many different combinations of $f_\theta$ and $h_\psi$ could produce the same observed $z_t$ path. Some regularization (e.g., encouraging $f_\theta$ to represent a *smooth, simple* resting tendency) is needed to keep the split meaningful rather than arbitrary.
- **Text embeddings are noisy.** Not all news that "sounds important" actually moves markets, and not all market-moving events are reported as news at the time.
- **A decreasing training loss is not the same as real predictive skill.** As with any model, EDS must be evaluated out-of-sample, on data and assets it was not trained on, with realistic transaction costs — a backtest that ignores costs and slippage can look far better than a live strategy ever would.
- **This is a research prototype.** It has no portfolio-level position sizing, no risk controls, and no execution model — those are separate, necessary layers on top of the probability $p_t$ before anything resembling live trading could be considered.

---

## 12. Quick Reference: All Formulas in One Place

| # | Formula | Meaning in one line |
| --- | --- | --- |
| 1 | $r_t = \frac{C_t}{C_{t-1}} - 1$ | Daily return |
| 2 | $y_{t+1} = \mathbf{1}\{r_{t+1} > 0\}$ | Up/down label |
| 3 | $\frac{dz_t}{dt} = f_\theta(z_t) + h_\psi(z_t, v_t)$ | Equilibrium dynamics: self-drift + news forcing |
| 4 | $z_{t+\Delta} \approx z_t + \Delta[f_\theta(z_t) + h_\psi(z_t, v_t)]$ | One numerical integration step |
| 5 | $z_0 = \mu_\phi(x_{t-L:t})$ | Encoder: recent prices → starting equilibrium estimate |
| 6 | $d_t = h_\psi(z_t, v_t)$ | Dissipation (shock) vector |
| 7 | $D_t = \lVert d_t \rVert^2$ | Dissipation energy (scalar) |
| 8 | $e_t = x_t - z_t$ | Deviation from equilibrium |
| 9 | $p_t = \sigma(a_\omega(z_t, e_t, D_t))$ | Final up/down probability |
| 10 | $\mathcal{L}_{\text{recon}} = \frac{1}{N}\sum (x_t - \text{dec}_\psi(z_t))^2$ | Reconstruction loss |
| 11 | $\mathcal{L}_{\text{pred}} = -\frac{1}{N}\sum[y\log p + (1-y)\log(1-p)]$ | Prediction loss (cross-entropy) |
| 12 | $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda\mathcal{L}_{\text{pred}}$ | Combined training objective |

---

## References

1. Chen, T. Q. et al., [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366).
2. Rubanova, Y. et al., [Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907).
3. Devlin, J. et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (representative text-embedding approach).
4. Niculescu-Mizil, A. and Caruana, R., [Predicting Good Probabilities with Supervised Learning](https://dl.acm.org/doi/10.1145/1102351.1102430).
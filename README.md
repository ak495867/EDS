# EDS — Equilibrium Dissipation Signal

Research prototype modeling financial markets as non-equilibrium dynamical systems: a slow-moving latent **equilibrium state** and a fast-reacting **observed state**, with the gap between them (the *dissipation signal*) used as the primary predictive feature.

**Hypothesis:** the most profitable moments are not when the market is calm, but during the transient phase as it dissipates the "shock" of new information.

## Status

Research prototype. No transaction costs, slippage, execution modeling, or live risk controls. Not investment advice.

## Contents

- [`MATH.md`](./MATH.md) — full mathematical formulation, explained from first principles (equilibrium dynamics, the dissipation ODE, loss function, evaluation methodology, worked numeric example)

## Approach at a glance

1. Encode raw market data into a latent embedding
2. Track a slow-moving equilibrium state (EMA over embeddings)
3. Evolve an observed state via a Neural ODE with a restoring force + information-impulse term
4. Derive the dissipation signal and its rate of change
5. Predict return / direction / volatility from the dissipation signal
6. Size positions via a Kelly-style rule, subject to standard risk constraints

## Evaluation

Benchmarked against Linear Regression, XGBoost, LSTM, and Transformer baselines on identical features, using purged walk-forward validation, Information Coefficient, Sharpe ratio, and significance testing (Diebold-Mariano, permutation tests).

## License

TBD.

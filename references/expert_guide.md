# Summary: "An Expert's Guide to Training PINNs" (Wang et al., 2023)

Reference: Sifan Wang, Shyam Sankaran, Hanwen Wang, Paris Perdikaris. "An Expert's Guide to Training Physics-Informed Neural Networks." arXiv:2308.08468v1, Aug 2023.

---

## Table of Contents

1. [Three Main Training Pathologies](#1-three-main-training-pathologies)
2. [The Core Training Pipeline (Algorithm 1)](#2-the-core-training-pipeline)
3. [Non-Dimensionalization (Section 3)](#3-non-dimensionalization)
4. [Network Architecture (Section 4)](#4-network-architecture)
5. [Training Algorithms (Section 5)](#5-training-algorithms)
6. [Miscellaneous Best Practices (Section 6)](#6-miscellaneous-best-practices)
7. [Benchmark Results and Ablation Studies (Section 7)](#7-benchmark-results)
8. [Hyperparameter Reference Tables](#8-hyperparameter-reference-tables)
9. [Implementation Notes](#9-implementation-notes)

---

## 1. Three Main Training Pathologies

Wang et al. identify three fundamental pathologies that prevent PINNs from converging to accurate solutions:

### 1.1 Spectral Bias
Neural networks are biased toward learning low-frequency functions first. High-frequency components converge exponentially slower. This is analyzed through the lens of Neural Tangent Kernel (NTK) theory:
- NTK eigenvalues decay rapidly with frequency index
- The convergence rate of each eigencomponent is proportional to its corresponding eigenvalue
- Result: PINNs produce blurry, over-smoothed solutions missing fine structures

**Fix**: Fourier feature embeddings map inputs to high-frequency signals before the MLP, controlling the NTK eigenspectrum.

### 1.2 Unbalanced Back-Propagated Gradients
The gradient magnitudes of PDE residual loss, BC loss, and IC loss can differ by orders of magnitude. The optimizer biases toward minimizing the term with largest gradients, causing other constraints to be poorly satisfied.

**Fix**: Gradient-norm or NTK-based adaptive loss weighting to equalize gradient magnitudes.

### 1.3 Causality Violation
Conventional PINNs minimize all PDE residuals simultaneously. They tend to first minimize residuals at later times, even before learning correct solutions at earlier times. This violates the causal structure of time-dependent PDEs.

**Fix**: Causal training — weight temporal chunks by cumulative residual from earlier times, forcing the network to learn sequentially along the time axis.

---

## 2. The Core Training Pipeline

Algorithm 1 from the paper specifies:

1. **Non-dimensionalize** the PDE system so all variables are O(1)
2. **Represent solution** with an MLP using Fourier feature embeddings and random weight factorization (tanh activation, Glorot init)
3. **Formulate weighted loss**: `L(θ) = λ_ic·L_ic + λ_bc·L_bc + λ_r·L_r`
   - PDE residual loss is split into M temporal chunks: `L_r = (1/M) * Σ w_i * L^i_r`
4. **Initialize all weights** `λ_ic = λ_bc = λ_r = 1`, `w_i = 1`
5. **Training loop** for S steps:
   - Sample collocation points randomly
   - Compute temporal weights: `w_i = exp(-ε * Σ(k=1 to i-1) L^k_r)`
   - Every f steps: compute gradient-norm weights λ̂, update with moving average: `λ_new = α·λ_old + (1-α)·λ̂_new`
   - Update θ via gradient descent

**Default hyperparameters**: f=1000, α=0.9, ε=1.0, gradient clipping recommended.

---

## 3. Non-Dimensionalization

### Why it matters

1. **Network initialization**: Glorot/Xavier scheme assumes inputs ~ N(0,1). Unscaled physical variables (e.g., x ∈ [0, 10⁶]) break this assumption.
2. **Disparity in variable scales**: Different physical quantities (pressure ~ 10⁵, velocity ~ 1) create unbalanced contributions.
3. **Convergence**: Unscaled variables force optimizer to take inconsistent step sizes across dimensions.

### How to do it

1. Choose characteristic scales: length L*, velocity U*, time T*, pressure ρ·(U*)², etc.
2. Transform: x* = x/L*, t* = t/T*, u* = u/U*, p* = p/(ρ·U*²)
3. Substitute into PDE to get dimensionless equations (involving Re, Pe, etc.)
4. Variables should all be O(1) after transformation

### Empirical evidence

Stokes flow ablation study: removing non-dimensionalization increased relative L2 error from 5.41×10⁻⁴ to 9.74×10⁻¹ (a 1800× increase).

---

## 4. Network Architecture

### 4.1 MLP Backbone
- 4-5 hidden layers, 128-256 neurons per layer
- **tanh** activation (infinitely differentiable, bounded output)
- **Never ReLU**: second derivative is zero, kills PDE residual gradients
- GeLU and sin are acceptable alternatives
- Glorot (Xavier) initialization

### 4.2 Fourier Feature Embeddings
```
γ(x) = [cos(Bx), sin(Bx)],  B ~ N(0, σ²)
```
- σ controls which frequencies the network is biased toward
- σ ∈ [1, 10] is recommended; match σ to the expected frequency content of the solution
- Too small σ → blurry predictions
- Too large σ → salt-and-pepper artifacts
- **This is the single most impactful technique.** Removing it from Allen-Cahn increased error from 5.84×10⁻⁴ to 4.35×10⁻¹ (745× worse).

### 4.3 Random Weight Factorization (RWF)
```
W = diag(exp(s)) · V,  s ~ N(µ, σI)
```
- Provides per-neuron adaptive learning rates
- Effectively shortens distance to minima in the factorized parameter space (proven in Theorem B.1)
- Recommended: µ = 0.5 or 1.0, σ = 0.1
- Applies to all layers; train s, V, b directly

### 4.4 Modified MLP
Two input encoders U(x), V(x) merged into each hidden layer:
```
f^(l)(x) = W^(l) · g^(l-1)(x) + b^(l)
g^(l)(x) = σ(f^(l)(x)) ⊙ U + (1 - σ(f^(l)(x))) ⊙ V
```
- Adds ~2× parameters but significantly better for nonlinear PDEs
- On Kuramoto-Sivashinsky: replacing modified MLP with standard MLP increased error from 1.42×10⁻⁴ to 2.98×10⁻³ (21× worse)

---

## 5. Training Algorithms

### 5.1 Causal Training
- Split temporal domain [0,T] into M equal chunks
- PDE residual loss per chunk: L^i_r
- Weight: w_i = exp(-ε · Σ(k=1 to i-1) L^k_r)
- ε=1.0 is the default "causal tolerance"
- Weights are detached (no backprop through w_i)
- If min(w_i) does not reach 1.0 by end of training, reduce ε

**Motivation**: The residual loss at time t_i should only be minimized after the solution at earlier times is learned. Without this, PINNs converge to wrong solutions.

### 5.2 Loss Balancing

**Gradient-Norm Scheme (recommended default)**:
- Compute gradient norm of each loss term w.r.t. θ
- Set λ̂ such that ||λ̂_ic · ∇L_ic|| = ||λ̂_bc · ∇L_bc|| = ||λ̂_r · ∇L_r||
- Update via moving average with α=0.9
- Update every f=1000 iterations

**NTK-Based Scheme (alternative)**:
- Use trace of NTK submatrices: λ̂_ic = (Tr(K_ic) + Tr(K_bc) + Tr(K_r)) / Tr(K_ic)
- More stable than gradient-norm but higher compute cost
- Use only diagonal elements of NTK for efficiency

### 5.3 Curriculum Training

**Temporal domain decomposition**: Divide [0,T] into windows, train sequentially. Each window's IC comes from the previous window's prediction at t_end.

**Parameter continuation**: Start with easier parameters (low Re, large viscosity), gradually increase difficulty. On lid-driven cavity at Re=3200: trained with increasing Re sequence [100, 400, 1000, 3200].

**When needed**: Chaotic systems (Kuramoto-Sivashinsky), high Reynolds number flows, long-time integration.

---

## 6. Miscellaneous Best Practices

### 6.1 Optimizer and Learning Rate
- **Adam**: consistently best, minimal tuning needed
- **lr = 0.001**, exponential decay at rate 0.9 every 2000 steps
- **NO weight decay** for forward problems (degrades accuracy)
- L-BFGS fine-tuning after Adam is beneficial but optional

### 6.2 Random Sampling
- Randomly sample collocation points every iteration
- Batch size: 4096-8192
- **Never use full-batch gradient descent** — it overfits PDE residuals
- Random sampling introduces beneficial regularization

### 6.3 Hard Constraints for Boundary Conditions

**Periodic BCs via Fourier embedding**:
- 1D: v(x) = [cos(2πx/P), sin(2πx/P)]
- 2D: v(x,y) = [cos(2πx/Px), sin(2πx/Px), cos(2πy/Py), sin(2πy/Py)]
- Time periodicity: add trainable P_t parameter

**Dirichlet/Neumann**: modify network output as `u = g_boundary + d(x) · network(x)` where d(x) is the distance to the boundary.

### 6.4 Modified MLP
Already described in Section 4.4 — prefer for complex nonlinear problems.

---

## 7. Benchmark Results

### Ablation Study Summary

For each benchmark, each component (Fourier Features, RWF, Grad Norm weighting, Causal training, Modified MLP, Non-dimensionalization) was individually disabled to measure its impact.

| Benchmark | Full Pipeline Error | Worst Ablation | Factor Worse |
|-----------|--------------------|-----------------|--------------|
| Allen-Cahn | 5.84×10⁻⁴ | No Fourier: 0.435 | 745× |
| Advection (c=80) | 1.02×10⁻² | No Grad Norm: 1.13 | 111× |
| Stokes flow | 5.41×10⁻⁴ | No Fourier: 0.956 | 1767× |
| Kuramoto-Sivashinsky | 1.42×10⁻⁴ | No Fourier: 1.86×10⁻² | 131× |
| Lid-driven cavity | ~10⁻¹ | No Modified MLP: 5.48×10⁻¹ | ~5× |

### State-of-the-Art Results (Best Models After Hyperparameter Tuning)

| Benchmark | Relative L2 Error |
|-----------|-------------------|
| Allen-Cahn | 5.37×10⁻⁵ |
| Advection (c=80) | 6.88×10⁻⁴ |
| Stokes flow | 8.04×10⁻⁵ |
| Kuramoto-Sivashinsky | 1.61×10⁻¹ |
| Lid-driven cavity (Re=3200) | 1.58×10⁻¹ |
| Navier-Stokes in torus | 2.45×10⁻¹ |

### Key Takeaways from Ablation Studies

1. **Fourier features are the most critical component.** Disabling them caused the largest error increase in every benchmark.
2. **Causal training is essential for time-dependent PDEs.** Without it, PINNs learn later times first (causality violation).
3. **Grad norm weighting prevents one loss term from dominating.** On advection equation, disabling it caused complete failure (error > 1.0).
4. **Modified MLP matters most for nonlinear problems.** On Kuramoto-Sivashinsky and lid-driven cavity, it was critical.
5. **Non-dimensionalization is non-negotiable for multi-scale problems.** On Stokes flow, disabling it was catastrophic.
6. **All components contribute positively.** Every ablation showed worse performance when any component was removed.

---

## 8. Hyperparameter Reference Tables

### Allen-Cahn (Best Model)
| Parameter | Value |
|-----------|-------|
| Architecture | Modified MLP |
| Layers / Width | 4 / 256 |
| Activation | Tanh |
| Fourier σ | 2.0 |
| RWF | µ=0.5, σ=0.1 |
| LR / Decay | 0.001 / decay 0.9 every 5000 steps |
| Training steps | 300,000 |
| Batch size | 8,192 |
| Weighting | NTK |
| Causal ε | 1.0, M=32 |

### Advection (c=80, Best Model)
| Parameter | Value |
|-----------|-------|
| Architecture | Modified MLP |
| Layers / Width | 4 / 256 |
| Activation | Tanh |
| Fourier σ | 1.0 |
| RWF | µ=1.0, σ=0.1 |
| LR / Decay | 0.001 / decay 0.9 every 2000 steps |
| Training steps | 200,000 |
| Batch size | 8,192 |
| Weighting | Grad Norm |
| Causal ε | 1.0, M=32 |
| Note | Time periodicity enforced via Fourier embedding |

### Stokes Flow (Best Model)
| Parameter | Value |
|-----------|-------|
| Architecture | Modified MLP |
| Layers / Width | 4 / 256 |
| Activation | GeLU |
| Fourier σ | 10.0 |
| RWF | µ=0.5, σ=0.1 |
| LR / Decay | 0.001 / decay 0.9 every 2000 steps |
| Training steps | 100,000 |
| Batch size | 8,192 |
| Weighting | Grad Norm |

### Kuramoto-Sivashinsky (Best Model)
| Parameter | Value |
|-----------|-------|
| Architecture | Modified MLP |
| Layers / Width | 5 / 256 |
| Activation | Tanh |
| Fourier σ | 1.0 |
| RWF | µ=0.5, σ=0.1 |
| LR / Decay | 0.001 / decay 0.9 every 2000 steps |
| Time windows | 10 |
| Steps per window | 200,000 |
| Batch size | 4,096 |
| Weighting | Grad Norm |
| Causal ε | 1.0, M=16 |

### Lid-Driven Cavity (Re=3200, Best Model)
| Parameter | Value |
|-----------|-------|
| Architecture | Modified MLP |
| Layers / Width | 5 / 256 |
| Activation | Tanh |
| Fourier σ | 10.0 |
| RWF | µ=1.0, σ=0.1 |
| LR / Decay | 0.001 / decay 0.9 every 10000 steps |
| Curriculum Re | [100, 400, 1000, 3200] |
| Steps per Re | [50000, 50000, 100000, 500000] |
| Batch size | 8,192 |
| Weighting | Grad Norm |

### Navier-Stokes in Torus (Best Model)
| Parameter | Value |
|-----------|-------|
| Architecture | Modified MLP |
| Layers / Width | 4 / 256 |
| Activation | Tanh |
| Fourier σ | 1.0 |
| RWF | µ=0.5, σ=0.1 |
| LR / Decay | 0.001 / decay 0.9 every 2000 steps |
| Time windows | 5 |
| Steps per window | 150,000 |
| Batch size | 8,192 |
| Weighting | Grad Norm |
| Causal ε | 1.0, M=16 |

### Navier-Stokes Around Cylinder (Best Model)
| Parameter | Value |
|-----------|-------|
| Architecture | Modified MLP |
| Layers / Width | 5 / 256 |
| Activation | Tanh |
| Fourier σ | 1.0 |
| RWF | µ=1.0, σ=0.1 |
| LR / Decay | 0.001 / decay 0.9 every 2000 steps |
| Time windows | 10 |
| Steps per window | 200,000 |
| Batch size | 8,192 |
| Weighting | Grad Norm |
| Causal ε | 1.0, M=16 |

---

## 9. Implementation Notes

### Diagnostic Tools from the Paper

1. **Gradient histograms**: Plot distributions of ∂L_ic/∂θ and ∂L_r/∂θ at training end. If one dominates, loss balancing is needed.
2. **Temporal PDE residual L_r(t)**: Plot PDE loss vs. time. If later times have lower loss than earlier times → causality violation.
3. **NTK eigenvalue spectrum**: Rapid decay indicates spectral bias → need Fourier features.
4. **min(w_i) over time**: Should converge to 1.0. If not → reduce causal tolerance ε.

### Performance Notes

- The paper uses JAX with multi-GPU data-parallel training (up to 256 GPUs)
- PyTorch implementations achieve similar results with same hyperparameters
- Random sampling + Adam is fast per iteration; cost is dominated by gradient computation through the network
- The additional cost of gradient-norm weight updates and causal weights is negligible

### Code Reference

The authors' JAX implementation is available at: https://github.com/PredictiveIntelligenceLab/jaxpi

---

## Summary Checklist

When building a PINN following Wang et al.'s methodology:

- [ ] PDE system non-dimensionalized (all variables O(1))
- [ ] Input coordinates normalized to [-1, 1]
- [ ] Fourier feature embeddings with σ tuned to problem
- [ ] Random weight factorization (µ=1.0, σ=0.1)
- [ ] MLP or Modified MLP with tanh activation (NEVER ReLU)
- [ ] Gradient-norm loss balancing enabled
- [ ] Causal training enabled (for time-dependent PDEs)
- [ ] Random sampling at each iteration
- [ ] Adam optimizer, lr=0.001, exponential decay
- [ ] Hard constraints for BCs when possible
- [ ] Curriculum training for stiff/long-time/chaotic problems
- [ ] Monitor per-component losses, λ weights, and min(w_i)

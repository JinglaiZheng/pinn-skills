---
name: pinn-skill
description: >
  Build production-quality Physics-Informed Neural Network (PINN) code using PyTorch.
  Use this skill whenever the user mentions PINN, physics-informed neural networks,
  physics-informed machine learning, scientific machine learning (SciML), solving PDEs
  with neural networks, or training difficulties with PINN models. Also trigger when the
  user wants to debug, improve accuracy of, or add features to an existing PINN model.
  Activate on any mention of PDE+neural network combinations, even if the user does not
  explicitly say "PINN."
---

# PINN Development Workflow

Follow these 8 steps in order to produce robust, high-accuracy PINN solutions.
Each step builds on the previous one. Do not skip steps.

---

## Step 1: Problem Analysis and Classification

Before writing any code, fully characterize the PDE problem:

### Identify the PDE type
- **ODE or PDE?** Spatial dimensions? Time-dependent or steady-state?
- **Classification**: elliptic, parabolic, hyperbolic, or mixed
- **Linearity**: linear vs. nonlinear (and nature of nonlinearity)
- **Order**: first-order, second-order, or higher

### Identify the domain and conditions
- Spatial domain Ω and its geometry (simple rectangle, complex, multi-scale)
- Boundary conditions: Dirichlet, Neumann, Robin, periodic, or mixed
- Initial conditions (for time-dependent problems)
- Known parameters (viscosity, diffusion coefficient, wave speed, etc.)

### Assess problem difficulty
- Does the solution contain shocks, sharp gradients, or discontinuities?
- Is the system stiff (e.g., high Reynolds number, small diffusion coefficient)?
- Is the behavior chaotic or sensitive to initial conditions?
- Are there known exact or analytical solutions?
- What is the timescale of interest (short vs. long-time integration)?

### Output
Produce a concise summary table listing: PDE form, domain, BC/IC types, known parameters, expected solution features, and difficulty factors. This drives all downstream decisions.

---

## Step 2: PINN Model Planning and Selection

Based on the problem analysis from Step 1, choose the model components. Do NOT write code yet — only specify choices and justify them.

### Network architecture
- **Default**: MLP with 4-5 hidden layers, 128-256 neurons per layer, `tanh` activation, Glorot initialization.
- **For complex nonlinear PDEs** (Navier-Stokes, Kuramoto-Sivashinsky): use **Modified MLP** (two input encoders U, V merged via element-wise multiplication in each hidden layer). See `references/pinn_techniques.md` for implementation.
- **Never use ReLU** — its second derivative is zero, which kills PDE residual gradients.
- **Width and depth**: 128-512 neurons, 3-6 hidden layers. Wider/deeper is not always better — too large networks become hard to optimize.

### Fourier feature embeddings
- **Always consider** Fourier feature embeddings to mitigate spectral bias.
- Map input coordinates to `[cos(Bx), sin(Bx)]` where `B ~ N(0, σ²)`.
- Scale factor `σ ∈ [1, 10]` — larger for problems with fine structures or high frequencies.
- This is the single most impactful technique for learning sharp gradients.

### Random weight factorization (RWF)
- Use as a drop-in replacement for standard dense layers.
- Factorizes each weight row as `w = diag(exp(s)) · v`.
- Recommended settings: `µ = 0.5` or `1.0`, `σ = 0.1`.
- Provides per-neuron adaptive learning rates at negligible cost.

### Loss weighting strategy
- **Default**: gradient-based self-adaptive weighting (recompute every ~1000 iterations).
- **Alternative**: NTK-based weighting (more stable but more expensive).
- Balance PDE residual loss, IC loss, and BC loss so their gradient norms are equal.

### Training strategy
- **Default**: Adam optimizer, initial lr = 0.001, exponential decay (rate 0.9, steps 2000-5000).
- **For time-dependent problems**: apply **causal training** — split temporal domain into M chunks, weight each chunk's PDE loss by how well earlier times are learned.
- **For stiff/long-time problems**: apply **curriculum training** — train sequentially on smaller time windows (time-marching) or gradually increase difficulty (e.g., increasing Reynolds number).
- **Fine-tuning**: after Adam, optionally run L-BFGS for 500-5000 iterations.

### Sampling strategy
- Use **random sampling** (not full-batch) at each iteration. Recommended batch sizes: 4096-8192.
- Full-batch gradient descent overfits PDE residuals and hurts generalization.
- For problems with localized features, consider adaptive (residual-based) sampling later.

### Boundary condition handling
- **Periodic BCs**: enforce exactly via Fourier feature embedding (hard constraint).
- **Dirichlet/Neumann/Robin**: consider hard constraints via distance functions or solution ansatz, otherwise soft constraints with proper loss weighting.

### Problem-specific method selection (informed by PINNacle benchmark)

The PINNacle benchmark (Hao et al., 2024) evaluated 10+ methods across 22 diverse PDEs. Use these evidence-based recommendations to match methods to problem types:

**Complex geometry problems** (e.g., heat conduction around obstacles, irregular domains):
- Top performers: **FBPINN** (domain decomposition), **PINN-NTK** (NTK-based loss balancing), **RAR** (adaptive resampling)
- Why: domain decomposition isolates geometric complexity per subdomain; NTK reweighting automatically focuses loss weight on hard regions

**Multi-scale problems** (e.g., solutions with widely varying length scales):
- Top performers: **MultiAdam** (scale-invariant optimizer), **FBPINN**, **PINN-LRA** (gradient-norm loss balancing)
- Why: MultiAdam adapts learning rates per parameter to handle disparate gradient scales

**Nonlinear / chaotic problems** (e.g., Kuramoto-Sivashinsky, Gray-Scott, Navier-Stokes):
- Top performers: **LAAF** (locally adaptive activation), **FBPINN**, **MultiAdam**
- Warning: **all methods struggle** on these problems — expect to combine multiple techniques and use curriculum training
- Important: some methods fail completely for certain cases (e.g., PINN-NTK fails on Kuramoto-Sivashinsky with L2RE > 95%)

**High-dimensional problems** (e.g., Poisson-Nd, Heat-Nd):
- Top performer: **L-BFGS** second-order optimizer (only method that solved Heat-Nd)
- Loss reweighting methods tend to overfit on high-dimensional PDEs

**Inverse problems** (e.g., coefficient reconstruction from noisy observations):
- Top performer: **vPINN** (variational formulation) — significantly outperforms all other methods
- **gPINN** (gradient-enhanced loss) also competitive

**Parameterized PDEs** (solving the same PDE class with varying parameters):
- Top performer: **PINN-NTK** — best at automatically adjusting loss weights across different parameter regimes

Full results and method details are in `references/pinnacle_benchmark.md`.

### Output
A model plan document listing all architecture and training choices with justifications tied to the problem analysis.

---

## Step 3: PINN Code Writing

Write the complete PINN code in PyTorch. Follow these rules strictly:

### Data preprocessing (MANDATORY)
- **Non-dimensionalize the PDE** before coding. Choose characteristic scales (length L*, velocity U*, time T*, etc.) and scale all variables to O(1).
- **Normalize input coordinates** to `[-1, 1]` or `[0, 1]`. This is non-negotiable — unnormalized inputs make Glorot initialization ineffective and cause training instability.

### Network implementation
- Implement the architecture chosen in Step 2 (MLP or Modified MLP).
- Include Fourier feature embedding as a pre-processing layer (not trainable, except the scale σ may be tuned).
- Apply RWF if selected: after Glorot init, factorize each weight matrix as `W = diag(exp(s)) · V` and register `s` and `V` as trainable parameters.
- Use `torch.autograd.grad` with `create_graph=True` for computing PDE derivatives. Do NOT use `torch.diff` or finite differences.

### Loss function
- Implement IC loss, BC loss, and PDE residual loss as separate terms.
- For causal training: split temporal domain into M chunks, compute per-chunk PDE loss, and weight by `w_i = exp(-ε * sum(L_r^k for k < i))`. Use `torch.no_grad()` or `.detach()` for the weights so they don't back-propagate.
- Implement the gradient-based loss weighting: every `f` iterations, compute `λ` from the gradient norms of each loss term, and update via moving average with `α = 0.9`.

### Training loop
- Adam optimizer, lr=0.001, exponential decay schedule.
- Randomly sample collocation points each iteration (batch_size = 4096-8192).
- Log individual loss components (IC, BC, PDE residual) every 100-500 iterations.
- Track gradient norms and loss weights for diagnostics.
- Optionally run L-BFGS after Adam for fine-tuning.

### Reproducibility
- Set random seeds for torch and numpy.
- Use `torch.manual_seed(seed)`.

### Output requirements
- A single self-contained `.py` file that runs end-to-end.
- Command-line output must show: device info, loss per component during training, final relative L2 error.
- Save the trained model checkpoint (`.pth` file).

---

## Step 4: Code Checking

Before running, verify the code mechanically:

### Checklist
1. **Input normalization**: Are all spatial and temporal coordinates normalized to [-1, 1] or [0, 1]?
2. **Non-dimensionalization**: Have PDE variables been scaled to O(1)?
3. **Derivatives**: Are all PDE derivatives computed with `torch.autograd.grad(..., create_graph=True)`?
4. **Activation**: Is the activation NOT ReLU? (tanh, sin, or GeLU is correct)
5. **Loss terms**: Is each loss term computed on its own set of collocation points?
6. **Detached weights**: Are causal temporal weights and loss-balancing weights detached from the computation graph?
7. **Boundary conditions**: Are the BCs correctly enforced (hard or soft)?
8. **Periodic BCs**: If applicable, are they enforced as hard constraints via Fourier embedding?
9. **Optimizer**: Is Adam being used (not SGD without momentum)?
10. **Random sampling**: Are collocation points resampled each iteration?
11. **No weight decay**: Weight decay degrades PINN accuracy for forward problems — ensure it's disabled.
12. **Gradient clipping**: Consider adding gradient clipping (max_norm=1.0) for stability.

### Run a quick sanity check
- Train for 100 iterations with a small network and batch size.
- Verify all loss components produce finite values.
- Verify gradients are non-zero and not exploding.

---

## Step 5: Reference Traditional Numerical Solver

Write a traditional numerical solver to produce a reference (ground truth) solution for comparison:

- **Finite difference**: for simple rectangular domains; use explicit/implicit schemes as appropriate, with sufficiently fine grid.
- **Finite element (FEniCS/Firedrake)**: for complex geometries.
- **Spectral methods**: for periodic problems with smooth solutions.
- **Method of characteristics**: for hyperbolic conservation laws.
- **Chebfun/ChebPy**: for 1D problems with smooth solutions.

The reference solver must be independent of the PINN code (different numerical method) so that it can serve as ground truth for error computation.

---

## Step 6: Results Analysis

After training completes and the reference solution is available:

### Error metrics
- Compute **relative L2 error**: `||u_pred - u_ref||_2 / ||u_ref||_2`.
- Compute **absolute L∞ error**: `max|u_pred - u_ref|`.
- For time-dependent problems: compute error vs. time to see if errors accumulate.

### Diagnostic plots
- Plot the loss history (total loss and per-component losses).
- Plot the evolution of loss weights (λ values) over training.
- For causal training: plot the minimum temporal weight min(w_i) — it should converge to 1.0.

### If accuracy is insufficient
- Compare PINN solution vs. reference visually to locate regions of largest error.
- Check if the loss converged properly (all components decreasing).
- Check if loss weights are balanced (gradient norms should be similar).
- Identify which training pathology is present (spectral bias, causality violation, unbalanced gradients).

---

## Step 7: Results Visualization

Generate publication-quality figures:

### Required plots
1. **Solution comparison**: PINN prediction vs. reference solution side-by-side (or overlaid).
2. **Absolute error field**: `|u_pred - u_ref|` as a heatmap/contour.
3. **Loss convergence curves**: all loss components vs. iteration (log scale).
4. **Loss weight evolution**: λ values vs. iteration.

### For time-dependent problems, additionally
5. **Snapshot comparison** at several time instants.
6. **Error vs. time** plot.

Use matplotlib with clear labels, legends, and colorbars. Save all figures as PNG files (300 DPI).

---

## Step 8: Model Improvement

If the model accuracy is insufficient (relative L2 error > target threshold), systematically improve it. PINNacle (Hao et al., 2024) showed that vanilla PINN only solves ~45% of PDE benchmarks successfully, but different methods help for different problem types. Combine the expert guide's general techniques (Wang et al., 2023) with PINNacle's problem-specific findings:

### Priority order for general improvement (from Wang et al.)

Apply in this order based on ablation study evidence:

**Priority 1: Fourier feature embeddings** — If not already used, add Fourier feature embeddings. Tune the scale factor σ — start at 1.0, try values up to 10.0. This had the largest impact in ablation studies (error reduction of 100× or more for problems with sharp gradients).

**Priority 2: Loss balancing** — If gradient norms of different loss terms differ by more than 10×, enable gradient-based (PINN-LRA, recommended default) or NTK-based loss balancing. Rerun and check if all λ values stabilize. PINN-NTK is more stable but computationally heavier.

**Priority 3: Causal training** — For time-dependent problems where error accumulates in later times, enable causal training with M=16-32 chunks and ε=1.0. Check that min(w_i) converges to 1.0.

**Priority 4: Modified MLP** — Replace the standard MLP with the Modified MLP architecture. This is especially impactful for nonlinear PDEs (2-10× error reduction).

**Priority 5: Curriculum training** — For stiff problems or long-time integration, use time-marching or parameter continuation.

**Priority 6: Hard boundary constraints** — Replace soft BC constraints with hard constraints.

**Priority 7: Hyperparameter tuning** — Increase width/depth, adjust LR schedule, increase training iterations (up to 300K), increase batch size. PINNacle found batch sizes beyond 2048-8192 yield diminishing returns.

**Priority 8: Adaptive sampling** — Use residual-based adaptive refinement (RAR) for problems with localized features.

### Problem-specific strategies (from PINNacle)

After applying general techniques, use these targeted strategies based on your problem type:

**Complex geometry** → Enable **FBPINN** (domain decomposition with domain-specific normalization). Divide the domain into subdomains, train subnetworks on each, stitch together. Also try **RAR** to focus collocation points near boundaries/obstacles.

**Multi-scale** → Try **MultiAdam** optimizer (parameter-wise scale-invariant). If solution has separated fast and slow scales, also try **curriculum training** from slow to fast.

**Nonlinear / chaotic** → Try **LAAF** or **GAAF** (adaptive activation functions). LAAF adds a trainable slope per neuron — simple drop-in with negligible overhead. If single-shot training fails, use **time-marching curriculum**. Note: expect residual errors even with best methods — the PINNacle benchmark shows all methods struggle on KS and NS equations.

**High-dimensional** → Switch optimizer to **L-BFGS**. The PINNacle benchmark found it was the only method that solved the high-dimensional Heat equation. Avoid overfitting: use fewer training epochs (5k-20k) with larger batch sizes.

**Inverse problems** → Use **vPINN** (variational / hp-VPINN formulation). This reformulates the loss in weak form and significantly outperforms all other methods on inverse coefficient reconstruction tasks.

**Parametric PDEs** → Use **PINN-NTK** for automatic loss weight adjustment across varying parameters simultaneously.

### Iterative refinement

After each improvement, rerun and compare against the reference solution. Iterate until the target accuracy is achieved. If no method achieves satisfactory accuracy, consider:
- Combining multiple complementary methods (e.g., FBPINN + NTK weighting + LAAF)
- Using traditional numerical preconditioning within the PINN framework
- Reconsidering whether the problem is fundamentally out of reach for current PINN methods (see PINNacle limitations discussion in `references/pinnacle_benchmark.md`)

---

## Quick Reference: Default Hyperparameters

| Parameter | Default Value |
|-----------|--------------|
| Architecture | MLP (or Modified MLP) |
| Hidden layers | 4-5 |
| Neurons per layer | 256 |
| Activation | tanh |
| Initialization | Glorot (Xavier) |
| Fourier feature σ | 1.0 - 10.0 |
| RWF µ, σ | µ=1.0, σ=0.1 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| LR decay | Exponential, rate=0.9, steps=2000 |
| Batch size | 4096 - 8192 |
| Weight update frequency f | 1000 |
| Moving average α | 0.9 |
| Causal tolerance ε | 1.0 |
| Number of time chunks M | 16 - 32 |
| Weight decay | 0 (disabled) |
| Gradient clipping | Optional, max_norm=1.0 |

---

## Reference Files

- **`references/pinn_techniques.md`**: Comprehensive catalog of PINN improvement techniques with implementation patterns, when to use each, and expected impact. Read this when selecting techniques in Step 2 or iterating in Step 8.
- **`references/expert_guide.md`**: Detailed summary of Wang et al. "An Expert's Guide to Training PINNs" (2023), including the full training pipeline (Algorithm 1), theoretical foundations, all ablation study results, and per-benchmark hyperparameter tables. Read this for deep understanding of why each technique works and what training pathologies to watch for.
- **`references/pinnacle_benchmark.md`**: Summary of Hao et al. "PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs" (NeurIPS 2024). Covers problem-specific method selection, 10+ method comparisons across 22 PDEs, and which methods work best for each challenge type (complex geometry, multi-scale, nonlinear, high-dim). Read this when matching methods to problem types in Step 2 or choosing targeted improvements in Step 8.

---

## Guiding Principles

1. **Non-dimensionalization is non-negotiable.** Unscaled variables break initialization assumptions and cause unbalanced gradients. Always scale your PDE first.

2. **Start simple, add complexity deliberately.** Begin with a standard MLP + Fourier features + grad-norm loss balancing. Add Modified MLP, RWF, causal training, and curriculum training only as needed. Each addition should be justified by a specific problem.

3. **Monitor per-component losses, not just total loss.** The total loss can decrease while individual components diverge. Track IC loss, BC loss, and PDE residual loss separately.

4. **Respect the physics.** Hard constraints are better than soft constraints when feasible. Causal training respects how information propagates in physical systems. The PDE structure should inform the model design.

5. **Random sampling beats full-batch.** It reduces memory, adds regularization, and prevents overfitting to PDE residuals. Always resample collocation points each iteration.

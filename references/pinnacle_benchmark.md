# Summary: "PINNacle: A Comprehensive Benchmark of PINNs" (Hao et al., NeurIPS 2024)

Reference: Zhongkai Hao, Jiachen Yao, Chang Su, Hang Su, Ziao Wang, Fanzhi Lu, Zeyu Xia, Yichi Zhang, Songming Liu, Lu Lu, Jun Zhu. "PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs." NeurIPS 2024 Track on Datasets and Benchmarks.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dataset: 22 PDE Cases](#2-dataset)
3. [Methods Benchmarked](#3-methods-benchmarked)
4. [Main Results](#4-main-results)
5. [Problem-Specific Recommendations](#5-problem-specific-recommendations)
6. [Hyperparameter Insights](#6-hyperparameter-insights)
7. [Parameterized PDE Results](#7-parameterized-pde-results)
8. [Key Takeaways for Practitioners](#8-key-takeaways)

---

## 1. Overview

PINNacle is the largest systematic benchmark of PINN methods to date. It evaluates 10+ state-of-the-art PINN variants across 22 diverse PDE cases, categorized by four core challenges:

| Challenge | Description | Example PDEs |
|-----------|-------------|-------------|
| Complex Geometry | Irregular domains with obstacles, holes | Poisson-CG, Heat-CG, NS-CG |
| Multi-Scale | Solutions varying over widely different scales | Poisson-MS, Heat-MS, Wave-MS |
| Nonlinear / Chaotic | Sensitive dependence, stiff dynamics | Burgers, NS, KS, Gray-Scott |
| High Dimensional | Curse of dimensionality (5+ dims) | Poisson-Nd, Heat-Nd |

---

## 2. Dataset

The 22 PDE cases span multiple domains:

| Category | Cases | Key Challenge |
|----------|-------|---------------|
| Burgers | 1D-C, 2D-C | Nonlinear |
| Poisson | 2d-C, 2d-CG, 3d-CG, 2d-MS | Complex geometry, multi-scale |
| Heat | 2d-VC, 2d-MS, 2d-CG, 2d-LT | Time-dependent, complex geo, long-time |
| Navier-Stokes | 2d-C, 2d-CG, 2d-LT | Nonlinear, complex geo, long-time |
| Wave | 1d-C, 2d-CG, 2d-MS | Second-order time derivative, periodic |
| Chaotic | GS (Gray-Scott), KS (Kuramoto-Sivashinsky) | Nonlinear + chaotic dynamics |
| High Dim | PNd (Poisson-Nd), HNd (Heat-Nd) | Curse of dimensionality |
| Inverse | PInv, HInv | Coefficient reconstruction from noisy data |

Reference solutions generated via COMSOL 6.0 (FEM) for complex geometry problems and Chebfun (spectral methods) for chaotic problems.

---

## 3. Methods Benchmarked

### 3.1 Vanilla Baselines
- **PINN**: Standard formulation (Raissi et al., 2019)
- **PINN-w**: PINN with larger boundary condition weights

### 3.2 Loss Reweighting / Resampling
- **PINN-LRA** (Wang et al., 2021): Gradient-norm based adaptive loss weighting. Recomputes λ values so all weighted loss gradients have equal L2 norm.
- **PINN-NTK** (Wang et al., 2022): Neural Tangent Kernel eigenvalue-based loss weighting. Balances convergence rates across loss terms.
- **RAR** (Lu et al., 2021): Residual-based Adaptive Refinement — periodically adds collocation points at locations of highest PDE residual.

### 3.3 Novel Optimizer
- **MultiAdam** (Yao et al., 2023): Parameter-wise scale-invariant optimizer. Resistant to domain scale changes. Adapts learning rates per parameter based on gradient statistics.

### 3.4 Novel Loss Functions
- **gPINN** (Yu et al., 2022): Gradient-enhanced PINN. Adds gradient matching terms to the loss: `L = L_PDE + λ * ||∇(PDE_residual)||²`.
- **vPINN** (Kharazmi et al., 2021): hp-Variational PINN. Uses weak/variational formulation with domain decomposition via hp-refinement. Test functions from Legendre polynomials.

### 3.5 Novel Architectures
- **LAAF** (Jagtap et al., 2020): Locally Adaptive Activation Functions. Each neuron has a trainable slope parameter `a`: `σ(a · x)`. Adds negligible overhead.
- **GAAF** (Jagtap et al., 2020): Globally Adaptive Activation Functions. One global trainable slope `a` shared across the network: `σ(a · x)`.
- **FBPINN** (Moseley et al., 2021): Finite Basis PINN — domain decomposition with overlapping subdomains, domain-specific normalization, and window functions for smooth stitching.

### 3.6 Second-Order Optimizer
- **LBFGS**: Line-search based quasi-Newton optimizer tested as an alternative to Adam.

---

## 4. Main Results

Results from Table 3 of the paper (L2 Relative Error). All models trained for 20,000 epochs with lr=0.001, repeated 3 times.

### Classification: Methods that "solve" the problem (L2RE < 10%):

| PDE | Best Method | L2RE |
|-----|------------|------|
| Burgers1d-C | PINN | 1.45E-2 |
| Burgers2d-C | PINN-NTK | 2.60E-1* |
| Poisson2d-C | PINN-NTK | 1.23E-2 |
| Poisson2d-CG | PINN-NTK | 1.43E-2 |
| Poisson3d-CG | PINN-NTK | 9.47E-2 |
| Heat2d-MS | PINN | 6.21E-2 |
| Heat2d-CG | LAAF | 2.39E-2 |
| NS2d-C | LAAF | 3.60E-2 |
| NS2d-CG | LAAF | 8.24E-2 |
| Chaotic GS | PINN-NTK / MultiAdam / LAAF / GAAF / FBPINN | 7.99-9.47E-2 |
| High Dim PNd | LBFGS | 4.67E-4 |
| High Dim HNd | LBFGS | 1.19E-4 |
| Inverse PInv | vPINN | 2.45E-2 |
| Inverse HInv | PINN-w | 5.26E-2 |

*Burgers2d-C: no method achieved <10%, best was 26%.

### Problems where ALL methods fail (L2RE ≈ 100%):

| PDE | Best L2RE | Note |
|-----|-----------|------|
| Poisson2d-MS | 5.90E-1 (MultiAdam) | Multi-scale diffusion |
| Heat2d-VC | 2.12E-1 (PINN-LRA) | Varying coefficients |
| Heat2d-LT | 9.99E-1 | Long-time integration |
| NS2d-LT | 9.70E-1 (LBFGS) | Long-time Navier-Stokes |
| Wave1d-C | 9.79E-2 (PINN-NTK) | Marginal |
| Wave2d-CG | 7.94E-1 (GAAF) | Complex geo wave |
| Wave2d-MS | 9.82E-1 (LAAF) | Multi-scale wave |
| KS (Kuramoto-Sivashinsky) | 9.57E-1 (PINN-NTK) | Chaotic |

### Method Rankings (averaged across all PDEs)

No single method dominates. Each shines in specific scenarios:

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| PINN-NTK | Best overall for loss balancing; strong on complex geometry, parametric PDEs | Fails on chaotic KS, computationally expensive |
| LAAF | Best adaptive activation; strong on nonlinear PDEs | No improvement on multi-scale or high-dim |
| FBPINN | Best for complex geometry + chaotic; best on GS | Hard to tune (subdomain size, overlap), fails on KS |
| vPINN | Best for inverse problems; strong on Wave | No clear improvement on forward problems |
| MultiAdam | Simple drop-in; helps on multi-scale | Does not significantly outperform other methods |
| RAR | Helps on complex geometry | Limited benefit on nonlinear/chaotic |
| gPINN | Competitive on some forward problems | Unstable on many PDEs |
| LBFGS | Best for high-dimensional PDEs | Converges to poor local minima on nonlinear problems |

---

## 5. Problem-Specific Recommendations

### Complex Geometry
**Best methods**: FBPINN > PINN-NTK > RAR > LAAF

Strategy:
1. First try **PINN-NTK** — automatically focuses loss weight on hard boundary regions
2. If insufficient, use **FBPINN** — divide domain into subdomains, each with its own network. Use domain-specific normalization for stability. Overlap subdomains by ~20%.
3. **RAR** as a complementary technique — add more collocation points near boundaries/obstacles

### Multi-Scale
**Best methods**: MultiAdam > FBPINN > PINN-LRA

Strategy:
1. First replace Adam with **MultiAdam** — parameter-wise scale-invariant, handles disparate gradient scales
2. If still struggling, add **FBPINN** — isolate scales into different subdomains
3. **PINN-LRA** for dynamic loss balancing across scale-separated regions

Note: Multi-scale problems remain challenging — even the best methods achieve L2RE ≈ 0.5-0.6 on Poisson2d-MS.

### Nonlinear / Chaotic
**Best methods**: LAAF > GAAF > FBPINN > MultiAdam

Strategy:
1. Enable **LAAF** — trainable slope per neuron adds expressiveness at negligible cost. On NS2d-C: error 3.60% vs 4.70% for vanilla PINN.
2. For long-time chaotic problems (KS, GS): use **curriculum training** with time-marching
3. Consider **FBPINN** for chaotic problems with spatial complexity

Critical warning: KS equation is essentially unsolved by current PINN methods — all methods produce ~100% error. This is a current research frontier.

### High Dimensional
**Best method**: LBFGS

Strategy:
1. **Switch from Adam to LBFGS** — the PINNacle benchmark found LBFGS was the ONLY method to solve high-dimensional Heat equation (L2RE = 1.19E-4 vs 0.361 for vanilla PINN)
2. Use larger batch sizes to get better gradient estimates in high-dimensional space
3. Avoid overfitting: fewer epochs (5k-20k), early stopping

### Inverse Problems
**Best methods**: vPINN >> gPINN > PINN-w

Strategy:
1. Use **vPINN** (variational formulation) — reformulates the PDE in weak form using test functions. On HInv: vPINN achieves 0.456 vs 1.57 for vanilla PINN.
2. **gPINN** as an alternative — adds gradient-matching regularization

### Parametric PDEs
**Best method**: PINN-NTK

Strategy:
1. Use **PINN-NTK** for automatic loss weight adjustment — achieved best results on 3/6 parametric PDE classes
2. Weight tuning varies per parameter; NTK adapts automatically

---

## 6. Hyperparameter Insights

From the PINNacle ablation studies:

### Batch Size
- Larger batch sizes → generally better (more accurate gradient estimates)
- Saturation point: ~2048-8192 (diminishing returns beyond)
- For GS and Poisson2d-C: batch size 8192 is sufficient

### Training Epochs
- More epochs → monotonically better accuracy
- Plateau: ~20k-80k epochs (problem-dependent)
- Most experiments used 20k epochs as a reasonable tradeoff

### Learning Rate
- lr=1e-3 or 1e-4 most stable
- lr=1e-2: frequent error spikes (unstable)
- lr=1e-5: too slow convergence
- **Step decay schedule** recommended for best stability

### Computational Cost
- Total benchmark: ~776 GPU hours on RTX 2080 Ti
- Can be completed in ~4 days on 8 GPUs
- Per-PDE training: ~35 GPU hours average

---

## 7. Parameterized PDE Results

When solving the same PDE with varying parameters:

| PDE Class | Best Method | Average L2RE |
|-----------|------------|--------------|
| Burgers-P (2d-C) | PINN-NTK | 4.13E-1 |
| Poisson-P (2d-C) | PINN-NTK | 1.66E-2 |
| Heat-P (2d-MS) | LAAF | 6.27E-2 |
| NS-P (2d-C) | vPINN | 3.76E-1 |
| Wave-P (1d-C) | PINN-NTK | 2.69E-1 |
| High dim-P (HNd) | MultiAdam | 2.22E-3 |

Key finding: **PINN-NTK is the strongest method for parametric PDEs** (best on 3/6 classes), likely because it automatically discovers the right loss weights for each parameter regime.

---

## 8. Key Takeaways

1. **No free lunch.** No single PINN variant dominates across all problem types. Method selection must be problem-driven.

2. **Loss reweighting is broadly useful.** PINN-NTK and PINN-LRA are the most generally applicable improvements and should be the first things to try.

3. **Domain decomposition (FBPINN) is powerful but tricky.** It excels on complex geometry and chaotic problems but requires careful tuning of subdomain size and overlap.

4. **Adaptive activations help nonlinear problems.** LAAF is a simple, low-cost improvement for nonlinear PDEs.

5. **LBFGS is essential for high dimensions.** Adam-based methods fail on high-dimensional PDEs; second-order optimization is necessary.

6. **Variational formulation wins inverse problems.** vPINN significantly outperforms all other methods on coefficient reconstruction tasks.

7. **Some problems remain unsolved.** All current PINN methods fail on: long-time Navier-Stokes, multi-scale Wave, Kuramoto-Sivashinsky, and long-time Heat. These are active research areas.

8. **Hyperparameters matter as much as method choice.** PINNacle found that batch size, learning rate schedule, and training epochs significantly impact results across all methods.

9. **Combining methods is underexplored.** The benchmark tested methods independently, but combining complementary approaches (e.g., FBPINN + NTK + LAAF) may yield further improvements.

---

## Citation

```bibtex
@inproceedings{hao2024pinnacle,
  title     = {PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs},
  author    = {Hao, Zhongkai and Yao, Jiachen and Su, Chang and Su, Hang and Wang, Ziao and Lu, Fanzhi and Xia, Zeyu and Zhang, Yichi and Liu, Songming and Lu, Lu and Zhu, Jun},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {Track on Datasets and Benchmarks}
}
```

Code and data: https://github.com/i207M/PINNacle

# PINN Improvement Techniques Catalog

Comprehensive reference of techniques to improve PINN accuracy and training stability.
Organized by category with implementation guidance, expected impact, and when to use each technique.

---

## Table of Contents

1. [Architecture Techniques](#1-architecture-techniques)
2. [Input Encoding Techniques](#2-input-encoding-techniques)
3. [Loss Function Design](#3-loss-function-design)
4. [Loss Balancing Strategies](#4-loss-balancing-strategies)
5. [Training Strategies](#5-training-strategies)
6. [Sampling Strategies](#6-sampling-strategies)
7. [Boundary Condition Handling](#7-boundary-condition-handling)
8. [Optimization and Regularization](#8-optimization-and-regularization)
9. [Diagnostics and Monitoring](#9-diagnostics-and-monitoring)

---

## 1. Architecture Techniques

### 1.1 Standard MLP

The default backbone. Multi-layer perceptron with tanh activation and Glorot (Xavier) initialization.

```
Recommended: 4-5 hidden layers, 128-256 neurons/layer
Activation: tanh (or GeLU for some problems)
Initialization: Glorot uniform
Input dim → [256] × 4 → output dim
```

**When to use**: Always start here. Sufficient for most problems with proper training.

### 1.2 Modified MLP (Wang et al. 2021)

Two input encoders U, V that merge via element-wise multiplication in each hidden layer. Provides enhanced representational capacity for complex nonlinear PDEs.

```python
class ModifiedMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.U_encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.V_encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        U = self.U_encoder(x)
        V = self.V_encoder(x)
        h = x  # or pass through initial layer
        for layer in self.hidden:
            z = layer(h)
            h = torch.tanh(z) * U + (1 - torch.tanh(z)) * V
        return self.output(h)
```

**Impact**: 2-10× error reduction for nonlinear PDEs (Kuramoto-Sivashinsky, Navier-Stokes).
**When to use**: When standard MLP struggles with complex nonlinear dynamics. Adds ~20% compute overhead.

### 1.3 Random Weight Factorization (RWF, Wang et al. 2022)

Factorize each weight row as `W = diag(exp(s)) · V`, training s and V separately. This provides per-neuron adaptive learning rates.

```python
class RWFLinear(nn.Module):
    def __init__(self, in_features, out_features, mu=1.0, sigma=0.1):
        super().__init__()
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        self.s = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.V)
        nn.init.normal_(self.s, mean=mu, std=sigma)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        W = torch.diag(torch.exp(self.s)) @ self.V
        return F.linear(x, W, self.bias)
```

**Impact**: Consistent improvement across all benchmarks (10-50% error reduction).
**When to use**: Almost always. Negligible computational cost. Use µ=1.0, σ=0.1 as default.

### 1.4 Multi-Head / Multi-Output Networks

Separate output heads for different solution components (e.g., u, v, p in Navier-Stokes). Heads share the hidden trunk but have independent output layers.

**When to use**: Multi-variable PDEs where outputs have different scales or character.

### 1.5 Spatiotemporal Decomposition

Factor the solution as `u(t,x) = Σ f_i(t) · g_i(x)` (separable representation). Reduces the problem to learning 1D functions.

**When to use**: Problems where separation of variables is a good approximation.

---

## 2. Input Encoding Techniques

### 2.1 Fourier Feature Embeddings (Tancik et al. 2020)

Map input coordinates to `[cos(Bx), sin(Bx)]` where `B ~ N(0, σ²)`. This is the single most impactful technique for mitigating spectral bias.

```python
class FourierFeature(nn.Module):
    def __init__(self, in_dim, num_features, sigma=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(num_features, in_dim) * sigma, requires_grad=False)
        # Alternative: self.B = torch.randn(...) as a buffer

    def forward(self, x):
        proj = x @ self.B.T  # (batch, num_features)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        # Output dim = 2 * num_features
```

**Impact**: Massive — reducing error from 0.435 to 0.000584 on Allen-Cahn (745× improvement).
**When to use**: ALWAYS. This should be the default. Tune σ ∈ [1, 10] — larger values help with sharp gradients but too large introduces artifacts. For problems with known frequency content, set σ to match.

### 2.2 Multi-Scale Fourier Features

Use multiple σ values or concatenate features from different scales to capture both low and high frequencies.

```python
scales = [1.0, 5.0, 10.0]
features = []
for s in scales:
    features.append(fourier_features(x, sigma=s))
x = torch.cat(features, dim=-1)
```

**When to use**: Problems with solutions containing features at widely separated spatial/temporal scales.

---

## 3. Loss Function Design

### 3.1 Standard MSE Composite Loss

`L = λ_ic * L_ic + λ_bc * L_bc + λ_r * L_r`

Each term is MSE over sampled collocation points. This is the default. Works well when combined with proper loss balancing.

### 3.2 Physics-Constrained Loss (Hard Constraints)

Instead of penalizing BC/IC violation, embed them exactly into the network architecture. This transforms the problem from constrained to unconstrained optimization.

**Example (Dirichlet BC u=g on boundary)**:
```python
def forward(self, x):
    d = distance_to_boundary(x)  # 0 at boundary, > 0 inside
    return g(x) + d * self.network(x)  # exactly satisfies BC
```

**Impact**: Eliminates BC loss term entirely, reducing optimization difficulty.
**When to use**: When BC satisfaction is critical and distance function is easy to compute.

### 3.3 Variational / Weak Form Loss

Use the weak formulation of the PDE (integral form) instead of the strong form. More robust for PDEs with low-regularity solutions.

**When to use**: Problems with discontinuous coefficients or non-smooth solutions where pointwise residuals fail.

### 3.4 Sobolev Training

Add gradient-matching terms to the loss: `L = L_PDE + α * ||∇u_pred - ∇u_true||²`. Requires gradient data.

**When to use**: When gradient data is available (e.g., from experiments or reference solutions).

---

## 4. Loss Balancing Strategies

### 4.1 Gradient-Norm Based (Wang et al. — Recommended Default)

Recompute λ values such that all weighted loss gradients have equal L2 norm:

```
λ̂_ic = (||∇_θ L_ic|| + ||∇_θ L_bc|| + ||∇_θ L_r||) / ||∇_θ L_ic||
λ̂_bc = (||∇_θ L_ic|| + ||∇_θ L_bc|| + ||∇_θ L_r||) / ||∇_θ L_bc||
λ̂_r  = (||∇_θ L_ic|| + ||∇_θ L_bc|| + ||∇_θ L_r||) / ||∇_θ L_r||
```

Update via moving average: `λ_new = α * λ_old + (1-α) * λ̂_new`, with α=0.9.
Update frequency: every 1000 iterations.

```python
if iteration % 1000 == 0:
    g_ic = torch.norm(torch.cat([p.grad.flatten() for p in ...]))
    # Compute by temporary backward or stored gradients
    lambda_ic = (g_ic + g_bc + g_r) / g_ic
    # ... same for bc, r
    lambda_ic = 0.9 * lambda_ic_old + 0.1 * lambda_ic
```

**Impact**: Prevents one loss term from dominating. Critical for problems where PDE scales differ from BC/IC scales.

### 4.2 NTK-Based (Wang et al. 2022)

Use Neural Tangent Kernel eigenvalues (or just the trace) to balance convergence rates:

```
λ̂_ic = (Tr(K_ic) + Tr(K_bc) + Tr(K_r)) / Tr(K_ic)
```

Where `K = J · J^T` and J is the Jacobian of network outputs w.r.t. parameters.
In practice, use only diagonal elements of the NTK to save computation.

**Impact**: Similar accuracy to gradient-norm but more stable weights. Higher computational cost.
**When to use**: When gradient-norm weights are too noisy.

### 4.3 Fixed Manual Weights

Set λ values manually based on domain knowledge.

**When to use**: Only for simple problems where you know the right scale. Generally not recommended.

### 4.4 Uncertainty-Based Weighting

Treat each loss as a Gaussian log-likelihood and learn the noise parameters:
`L = L_ic/(2σ²_ic) + L_bc/(2σ²_bc) + L_r/(2σ²_r) + log(σ_ic*σ_bc*σ_r)`

**When to use**: Alternative when gradient-based methods are unstable.

---

## 5. Training Strategies

### 5.1 Causal Training (Wang et al. 2022)

For time-dependent PDEs, split temporal domain into M chunks. Weight PDE loss in chunk i by how well earlier chunks are minimized:

```
w_i = exp(-ε * Σ(k=1 to i-1) L^k_r)
L_r = (1/M) * Σ(i=1 to M) w_i * L^i_r
```

Use M=16-32 chunks, ε=1.0 (causal tolerance).

```python
# Compute per-chunk PDE loss
chunk_losses = []
for i in range(M):
    t_mask = (t >= i*dt) & (t < (i+1)*dt)
    chunk_losses.append(pde_loss[t_mask].mean())

# Compute causal weights
cumsum_loss = torch.cumsum(torch.stack(chunk_losses[:-1]), dim=0)
w = torch.exp(-epsilon * cumsum_loss)
w = torch.cat([torch.ones(1, device=...), w])

# Weighted PDE loss
weighted_pde_loss = (w * torch.stack(chunk_losses)).mean()
```

**Impact**: Critical for time-dependent problems. On Allen-Cahn, removing causal training increased error from 5.84×10⁻⁴ to 1.59×10⁻³.

### 5.2 Curriculum Training / Time-Marching

Divide temporal domain into windows. Train PINN on window 1, use its prediction at t_end as IC for window 2, and so on.

**For time-dependent PDEs**: 5-10 time windows, train 100K-200K iterations per window.
**For parameter continuation**: Gradually increase Re, decrease viscosity, etc. during training.

**Impact**: Essential for long-time integration and chaotic systems. On Kuramoto-Sivashinsky, single-shot training fails for T>0.3.

### 5.3 Transfer Learning

Train on a simpler version of the problem (coarser mesh, lower Re, shorter time), then use those weights to initialize training on the target problem.

**When to use**: When direct training fails or for parameter sweeps.

### 5.4 Multi-Fidelity Training

Train with both low-fidelity (coarse) and high-fidelity (fine) data or PDE residuals. Low-fidelity provides global structure; high-fidelity refines details.

---

## 6. Sampling Strategies

### 6.1 Uniform Random Sampling (Recommended Default)

Randomly sample collocation points from uniform distribution over the domain. Resample at every iteration.

**Why**: Introduces regularization, prevents overfitting to fixed points, reduces memory.

### 6.2 Latin Hypercube Sampling (LHS)

Stratified sampling ensuring better coverage of the domain.

**When to use**: When domain has complex geometry and uniform sampling misses regions.

### 6.3 Sobol Sequences / Quasi-Monte Carlo

Low-discrepancy sequences providing more uniform coverage than random sampling.

**When to use**: Low-dimensional problems where coverage quality matters more than regularization.

### 6.4 Residual-Based Adaptive Refinement (RAR, Lu et al. 2021)

Periodically evaluate PDE residual on a dense grid. Add points with largest residual to the training set.

```python
if iteration % rar_freq == 0:
    residual = evaluate_pde_residual(dense_grid)
    _, top_indices = torch.topk(residual.abs(), k=num_new_points)
    new_points = dense_grid[top_indices]
    collocation_points = torch.cat([collocation_points, new_points])
```

**When to use**: Problems with localized features (shocks, boundary layers) that need more resolution in specific regions.

### 6.5 Importance Sampling

Sample proportionally to PDE residual magnitude or loss value.

**When to use**: When residual varies significantly across the domain.

---

## 7. Boundary Condition Handling

### 7.1 Soft Constraints (Penalty Method)

Add BC violation as a penalty term in the loss function.

```python
L_bc = torch.mean((u_pred_boundary - u_bc_true)**2)
```

**Pros**: Simple to implement. **Cons**: BCs are only approximately satisfied.

### 7.2 Hard Constraints via Solution Ansatz

Modify network output to exactly satisfy BCs:

```python
def forward(self, x):
    d = distance_to_boundary(x)  # 0 on boundary
    g = boundary_function(x)     # BC value
    return g + d * self.network(x)
```

For periodic BCs on [0, P]: use Fourier embedding `v(x) = [cos(2πx/P), sin(2πx/P)]`. Any network taking v(x) as input automatically satisfies periodicity.

**Impact**: Eliminates BC loss term, reduces total loss terms, improves accuracy.

### 7.3 Exact BC Imposition via Distance Functions

Compute the signed distance function to the domain boundary and use it to construct a solution ansatz that satisfies Dirichlet, Neumann, or Robin BCs exactly.

Reference: Sukumar & Srivastava (2021), Lu et al. (2021).

### 7.4 Periodic BCs via Fourier Embedding

For period P in 1D: `v(x) = [cos(2πx/P), sin(2πx/P)]`.
For period P_x, P_y in 2D: `v(x,y) = [cos(2πx/P_x), sin(2πx/P_x), cos(2πy/P_y), sin(2πy/P_y)]`.
For unknown time periodicity: `v(t,x) = [cos(2πt/P_t), sin(2πt/P_t), ...]` where P_t is a trainable parameter.

**Impact**: In advection equation (c=80), adding time periodicity reduced error from 7.37×10⁻¹ to 1.02×10⁻².

---

## 8. Optimization and Regularization

### 8.1 Adam + L-BFGS Two-Phase Training

Phase 1: Adam for 100K-500K iterations (handles noisy gradients, escapes bad local minima).
Phase 2: L-BFGS for 500-5000 iterations (fine-tunes, converges rapidly near minimum).

```python
# Phase 1: Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# ... train ...
# Phase 2: L-BFGS
optimizer = torch.optim.LBFGS(model.parameters(),
    lr=1.0, max_iter=5000, history_size=50,
    line_search_fn='strong_wolfe')
```

**When to use**: L-BFGS is especially helpful when Adam converges slowly near the minimum.

### 8.2 Learning Rate Scheduling

- **Exponential decay**: `lr = lr_0 * decay_rate^(step/decay_steps)`. Default: rate=0.9, steps=2000.
- **Cosine annealing**: `lr = lr_min + 0.5*(lr_0 - lr_min)*(1 + cos(π*step/T_max))`.
- **Cyclic LR**: Oscillate between lr_min and lr_max to escape local minima.
- **ReduceLROnPlateau**: Reduce LR when loss plateaus.

### 8.3 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**When to use**: When loss spikes or gradients explode. Especially useful during early training.

### 8.4 Weight Decay

**NOT recommended for forward PINN problems.** Weight decay degrades predictive accuracy. Use only for inverse problems with noisy data.

### 8.5 Exponential Moving Average (EMA) of Weights

Maintain a shadow copy of model parameters updated via EMA. Use the EMA model for inference.

```python
ema_model = copy.deepcopy(model)
for param, ema_param in zip(model.parameters(), ema_model.parameters()):
    ema_param.data = 0.999 * ema_param.data + 0.001 * param.data
```

---

## 9. Diagnostics and Monitoring

### Training Metrics to Track

1. **Per-component losses** (IC, BC, PDE residual): plot separately on log scale.
2. **Loss weights (λ)**: verify they stabilize, not diverge.
3. **Gradient norms**: verify balanced across loss components.
4. **NTK eigenvalues** (optional): check for spectral bias (rapid eigenvalue decay at high indices).
5. **Causal weights (min w_i)**: should converge to 1.0; if not, reduce ε.

### Common Failure Modes and Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| PDE loss decreases but BC/IC loss stays high | Unbalanced loss weights | Enable gradient-norm loss balancing |
| All losses decrease but solution is wrong | Causality violation | Enable causal training (for time-dependent) |
| Solution is blurry, misses fine features | Spectral bias | Add Fourier features, increase σ |
| Loss oscillates or diverges | Learning rate too high | Reduce lr or add gradient clipping |
| Loss stagnates at high value | Network too small or local minimum | Increase width, try L-BFGS, or curriculum training |
| High error at later times | Error accumulation | Time-marching / curriculum training |
| BCs not satisfied | Soft constraint too weak | Switch to hard constraints |
| Loss is NaN | Exploding gradients | Gradient clipping, reduce lr, check input scaling |

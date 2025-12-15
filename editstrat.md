To extend *this* Neural Geodesic Flows (NGF) implementation to a **pseudo-Riemannian** setting (indefinite metric, as in relativity), you mainly need to relax/replace one structural assumption that is baked into both the thesis and the code design:

* In the thesis, the learned chart metric is explicitly required to be **symmetric positive definite (SPD)** in coordinates (“any (g(x)) … has to be symmetric positive definite”). 
* In experiments, that SPD constraint is enforced by parameterizing (g_\theta) as (g_\theta = LL^{\top}) or (g_\theta = I + LL^{\top}). 
* The implementation outline computes Christoffel symbols (\Gamma) from (g) via autodiff and then integrates the geodesic ODE (RK4). 

A pseudo-Riemannian extension keeps the same high-level architecture ( \bar y = \phi_\theta \circ \exp_{g_\theta}(\cdot,t) \circ \psi_\theta(y)) , but changes how (g_\theta) is represented and regularized.

## What changes mathematically (small) vs. what changes numerically (big)

### What stays essentially the same

1. **Levi–Civita connection formula** still applies for any *non-degenerate* metric (pseudo-Riemannian included). So your pipeline “learn (g_\theta(x)) → compute (\Gamma(x)) → integrate geodesic ODE” remains valid. 
2. **Losses and training loop** (differentiate through ODE solver + autoencoder) are unchanged in form. 

### What becomes more delicate

1. You lose SPD guarantees: (g(x)) can be indefinite, and if it drifts toward **degeneracy** ((\det g \to 0)), computing (\Gamma) becomes unstable because it requires (g^{-1}).
2. “Sectional curvature” and related diagnostics need care: in pseudo-Riemannian geometry, some 2-planes are null/degenerate, and curvature scalars can behave differently; you must avoid divisions by near-zero Gram determinants (the code outline divides by (\det g_x) in a curvature computation). 

## The key engineering change: replace the SPD metric parameterization

Right now, the thesis’ standard choice (g_\theta = LL^{\top}) (or (I+LL^{\top})) **cannot** produce negative eigenvalues. 
To get a fixed signature ((p,q)) you want a factorization that enforces:

* symmetry,
* non-degeneracy,
* and exactly (p) “+” and (q) “−” directions everywhere.

Two practical parameterizations:

### Option A (recommended): fixed-signature “(L \Sigma L^{\top})” with positive scales

Let your network output:

* a lower-triangular matrix (L(x)) with ones (or unconstrained) on the diagonal,
* positive diagonal scales (s_i(x) > 0).

Define:
[
\Sigma(x)=\mathrm{diag}(s_1(x),\dots,s_p(x),-s_{p+1}(x),\dots,-s_{p+q}(x)),
\quad
g_\theta(x)=L(x),\Sigma(x),L(x)^{\top}.
]

Implementation detail: enforce (s_i(x)=\mathrm{softplus}(a_i(x))+\epsilon) so they’re bounded away from 0.

**Why this fits NGF well**

* Same interface as before: (g_\theta(x)) is an (m\times m) symmetric matrix field.
* Signature is guaranteed (as long as all (s_i>0)), which prevents the optimizer from “accidentally” switching signs.

### Option B: eigenvalue parameterization with signed spectrum

Have the network output an orthogonal-ish matrix (Q(x)) (hard to keep orthogonal without special layers) and signed eigenvalues. This is typically more expensive/fragile than Option A in JAX.

## Changes you will need in the NGF code path (moderate, localized)

Below is the concrete checklist relative to the thesis’ algorithmic outline. 

### 1) Metric network (g_\theta)

* Replace the SPD construction (LL^{\top}) / (I+LL^{\top}). 
* Add a **signature hyperparameter** ((p,q)) (e.g., Minkowski: (p=1,q=3) or (p=3,q=1) depending on convention).

### 2) Inversion / solving for (g^{-1})

Christoffels require (g^{k\ell}):
[
\Gamma^k_{ij}=\tfrac12 g^{k\ell}(\partial_i g_{j\ell}+\partial_j g_{i\ell}-\partial_\ell g_{ij}).
]
Your current “find (\Gamma) using autodiff on (g)” step still works, but you must ensure the code uses a numerically stable inverse/solve. 

Practical tweak: prefer `jax.scipy.linalg.solve(g, rhs)` over explicitly forming `inv(g)`.

### 3) Regularization terms (new, important)

With SPD metrics you can often “get away” with little regularization; indefinite metrics usually need explicit safeguards:

* Penalize near-degeneracy: e.g. add (\lambda \sum_x \mathrm{softplus}(\tau - \log|\det g(x)|)).
* Penalize extreme conditioning: e.g. log condition number or (|g^{-1}|) proxies.
* Optionally penalize signature violations if you do not hard-enforce them (Option A hard-enforces).

### 4) Curvature/diagnostic routines (if you use them)

If you compute sectional curvature like in the outline, add guards for null planes and small denominators. 
This does not affect training unless curvature terms enter your loss, but it will affect analysis plots.

### 5) Initialization

The thesis notes that the default initialization yields “straight line geodesics” and is a good start. 
For pseudo-Riemannian, you typically want to initialize (g_\theta) close to a **constant Minkowski metric** in your chosen chart, e.g. (\eta=\mathrm{diag}(-1,1,1,1)) or (\mathrm{diag}(1,-1,-1,-1)). With Option A, that is easy: start (L\approx I) and (s_i\approx 1).

## How much reworking is it?

If you scope it to “make NGF support pseudo-Riemannian metrics and still train end-to-end,” it is **moderate, not a rewrite**:

* **Low churn:** architecture, geodesic ODE integration, losses, and training loop stay the same.  
* **Medium churn:** metric parameterization + stable inverse/solve + regularization.
* **Potentially high churn (only if you go further):**

  * If you want *physical relativity semantics* (timelike/spacelike normalization constraints, proper time parametrization, enforcing causal structure, etc.), you will likely add constraints on the latent velocity components and/or modify the training data representation (e.g., learning geodesics parameterized by proper time rather than an arbitrary (t)).

If you tell me which signature you want (e.g. (1+3) vs (3+1)) and whether you care about timelike normalization (e.g. (g(\dot x,\dot x)=-1)), I can propose a minimal set of additional loss terms that makes the training behave more like GR, rather than “just” indefinite-metric geometry.

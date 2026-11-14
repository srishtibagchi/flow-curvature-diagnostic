# Flow curvature diagnostic

A reproducible starting point for comparing **independent-coupling conditional
flow matching (CFM)** and **optimal-transport flow matching (OT-FM)**, with
trajectory diagnostics separated from model-fitting error.

## Setup

Python 3.10+ and [uv](https://docs.astral.sh/uv/). The development interpreter is
pinned to Python 3.12 in `.python-version`.

```bash
uv sync --locked --extra notebook
```

For neural training, use `uv sync --locked --extra notebook --extra neural`.
The neural extra uses CPU PyTorch wheels to avoid unnecessary CUDA dependencies.
Dependencies live in `pyproject.toml`; `uv.lock` pins the resolved environment.
After intentionally changing dependencies, run `uv lock` and commit both files.
Choose this environment as your notebook kernel. Editable installation makes
`flow_curvature` importable from notebooks without `sys.path` modifications.

## Run

From the repository root:

```bash
# Fast, exact-field scientific control: no training or torch required.
uv run --locked flow-curvature --output-dir results/analytic-seed42

# Same MLP, optimizer, initial weights, samples, and integration grid;
# only the endpoint coupling changes.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked --extra neural flow-curvature --train-steps 2000 --output-dir results/neural-seed42

uv run --locked --extra neural pytest
```

Use a new output directory for each run; existing runs are never overwritten.
The default neural settings are a starting point, not a convergence guarantee.
Repeat with `--seed 43`, etc. to measure variability, and double
`--integration-steps` to assess discretization error. Defaults use CPU.

## Scientific baseline

The toy source is `N(0, I)` in 2D. The target has mean `(2, 1)` and covariance
`A A^T`, where `A` is the symmetric matrix with principal scales `(0.5, 1.8)`
rotated by `0.6` radians. Its population quadratic-cost OT map is `x1 = m + A x0`.
This deliberately simple, nonspherical example has closed-form marginal fields:

- **OT:** `v(t,x) = m + (A-I) ((1-t)I+tA)^(-1) (x-tm)`.
- **Independent:** with `C = A A^T`,
  `v(t,x) = m + [tC-(1-t)I] [(1-t)^2 I+t^2 C]^(-1) (x-tm)`.

Both use linear conditional paths and targets `u = x1-x0`. But integrating
their **marginal** fields gives different trajectories: population OT paths are
straight; independent-coupling paths generally curve. Plotting interpolated
training pairs alone cannot distinguish them: those paths are straight for both.

The analytic control should give OT bending energy effectively at roundoff and
nonzero independent bending. Focused tests enforce these numerical thresholds.
The optional neural experiment learns these objectives from finite data using a
time-conditioned MLP. OT pairing uses exact squared-distance **minibatch
assignment**, not a population OT oracle. Learned OT is therefore not guaranteed
to have zero curvature. OT-FM and independent CFM here differ in **coupling**,
not architecture; `VelocityMLP` is a replaceable architecture baseline.

## Diagnostic

For velocity `v` and acceleration `a` along a trajectory:

```text
a_perp = a - v * <v,a> / ||v||^2
kappa = ||a_perp|| / ||v||^2
bending_energy = integral kappa^2 ds
```

The main diagnostic is geometric bending energy; unlike raw acceleration it
does not penalize changing speed along a straight line. We also save
arc-length-weighted mean curvature, acceleration energy, path/chord lengths,
and absolute length excess. Curvature is scale-dependent, so compare methods
in the same coordinates. These statistics measure straightness, **not** whether
the target distribution was learned: a constant or collapsed flow can be straight.
Endpoint mean/covariance errors and Gaussian-moment W2 are reported alongside
curvature; the latter is not a full distribution metric for non-Gaussian outputs.

Central finite differences are evaluated at interior points of a uniform grid;
endpoint half-intervals are omitted from integrals. Samples with speed below
`1e-8` have undefined geometric curvature and are excluded, with
`stationary_fraction` reported explicitly. Wholly stationary paths have zero
bending by convention. Near-stationary paths can have large curvature: inspect
per-particle values and grid convergence, not only the aggregate mean.

## Layout and notebook use

### Verified seed-42 baseline

With the default dataset, 256 held-out particles, 200 RK4 steps, and 2,000
training steps (single-threaded CPU):

| Marginal flow | Median bending energy |
| --- | ---: |
| Analytic population OT | 5.56e-26 |
| Analytic independent CFM | 0.553 |
| Learned minibatch OT-FM | 0.000371 |
| Learned independent CFM | 0.587 |

Doubling the integration grid changes the learned medians by less than 2%.
These are a sanity check on one seed, not a general performance claim.
Independent-flow means are much larger than medians because near-stationary
trajectories can dominate bending energy.

### Project structure

```text
src/flow_curvature/
  data.py           seeded endpoints and Gaussian OT map
  coupling.py       independent and minibatch-OT endpoint pairing
  fields.py         exact marginal Gaussian velocity fields
  neural.py         optional time-conditioned MLP and CFM objective
  integration.py    shared NumPy velocity API and fixed-step RK4
  diagnostics.py    per-trajectory metrics and summaries
  experiment.py     reproducible CLI and persistence
  plotting.py       notebook-friendly trajectory plots
notebooks/          baseline walkthrough
data/processed/     generated endpoint arrays (gitignored)
results/            run outputs (gitignored)
tests/              geometric, scientific, persistence, and neural checks
```

Start Jupyter with `uv run --locked --extra notebook jupyter lab`, then open
[the baseline notebook](notebooks/01_gaussian_baseline.ipynb).

Each run saves `summary.json` (configuration, Gaussian parameters, independent
data/evaluation/model seeds, Python and dependency versions, aggregate metrics),
`trajectories.npz` (times and all paths), and per-method `*_metrics.npz` arrays.
Neural runs also save loss histories and model state dictionaries. Source/target
training sets and held-out initial points are saved under `data/processed/`.
No network access or downloads are required to run an installed experiment.
Use `uv sync --locked` to reconstruct the pinned dependencies. Runs are
deterministic within a fixed numerical environment, not necessarily bitwise
equivalent across platforms.

To extend the comparison, supply another callable `velocity(t, points)` to
`integrate`, or train another model with the same coupling/data/evaluation
protocol. Treat architecture, coupling, path formulation, and solver resolution
as separate experimental axes.

## Background

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482)

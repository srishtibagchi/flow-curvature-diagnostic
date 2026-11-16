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

## Image representation pilot

The optional image experiment compares **DINOv2, SigLIP2, and MAE RAEs** from
[NYU Vision-X](https://huggingface.co/collections/nyu-visionx/rae),
`stabilityai/sd-vae-ft-mse`, and pixels without an autoencoder. It trains a
separate **tiny DiT** for independent and minibatch OT flow matching in each
space: ten controlled models, not ten released pretrained diffusion models.
Both encoders and decoders remain frozen. SD-VAE uses the deterministic
posterior mean; RAEs use deterministic unmasked encoding with no added noise.

```bash
uv sync --locked --extra image

# Inspect exact immutable checkpoint revisions, files, bytes, and disk estimates.
# This command does not download datasets or weights.
uv run --locked --extra image flow-curvature-images --manifest

# Only after approving that manifest: fetch the listed files and CIFAR-10.
uv run --locked --extra image flow-curvature-images --download --download-only

# Offline execution from downloaded assets; use a new output directory.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --locked --extra image \
  flow-curvature-images --output-dir results/images-seed42

uv run --locked --extra image pytest tests/test_images.py
```

Without `uv`, an already prepared environment can run
`python -m flow_curvature.image_experiment` with the same arguments.
No dataset/model download occurs without the explicit `--download` flag.
The selected full frozen model/config payloads total approximately **6.41 GB**;
the CIFAR-10 archive adds **170.5 MB** before cache reuse. RAE weight licenses
are unspecified in their model cards; the upstream software's MIT license
does not establish a weight license. Review access and license requirements
before downloading. No released Stage-2 DiT checkpoint is downloaded.

### Protocol and limitations

Defaults use seed 42, 256 CIFAR-10 training images, 2 images from its separate
test split, 256×256 bicubic rendering, 200 training updates, batch size 8,
learning rate `1e-3`, and 8 fixed RK4 steps. CIFAR-10 upscaling and this training
budget are a **CPU pipeline smoke test**, not a converged image-generation or
ImageNet representation benchmark. Cite Alex Krizhevsky's *Learning Multiple
Layers of Features from Tiny Images* (2009) when using CIFAR-10.

The DiT-style backbone uses adaptive LayerNorm continuous-time conditioning
(smooth Fourier features on `[0,1]`, not integer diffusion timesteps), width 64,
depth 2, four attention heads, and 256 spatial tokens. RAE patches are 1×1,
SD-VAE patches 2×2, and pixel patches 16×16. Input/output projections differ;
backbone and total parameter counts are saved rather than claiming identical
full architectures or compute. Within a representation, both couplings share
initial weights, raw endpoint batches, time draws, and evaluation noise;
hashes verify that control. Noise shapes differ across representations.
OT cost is measured in each representation's normalized training coordinates.
In particular, the 64-wide input projection compresses 768 features per RAE
or pixel patch, whereas VAE patches have only 16 features. This is an unequal
information bottleneck, not a capacity-matched scientific comparison. A larger
GPU benchmark should address this architectural confound explicitly.

Checkpoint preprocessing and latent statistics are respected. An additional
fixed global scalar mean/std is fitted on training targets only and inverted
before native-coordinate measurement and decoding. Native curvature depends
on those coordinates and dimensionality; **do not rank raw native bending
energies as if their units were identical**.

The runner measures full-dimensional learned ODE paths in checkpoint-native
coordinates and continuous decoded 256×256 RGB coordinates. Pixel paths use
identity decoding. It also reports:

- `normalized_bending = bending_energy * path_length`, invariant to a uniform
  scalar coordinate rescaling, not anisotropic maps or nonlinear decoding;
- `relative_length_excess = path_length / chord_length - 1`, with explicit
  undefined counts for zero or roundoff-scale chords;
- RMS-coordinate path length, curvature distributions, and stationary fraction;
- decoded straight-native-path controls to expose decoder-induced bending;
- representative same-model 8→16-step refinement, with warnings for changes
  exceeding 5% (not a convergence guarantee);
- held-out reconstruction MSE/PSNR, endpoint diversity, image grids, and an
  exploratory common-pixel two-sample MMD using a training-defined bandwidth.

Generated endpoints are **not paired reconstructions** of held-out images.
Reconstruction PSNR is separate from generative quality. MMD with 2 samples
is not FID; negative unbiased MMD² estimates are valid. Straightness alone can
favor collapsed or undertrained models. Early decoded noise may be off-manifold
and differ substantially between representations. Clipping/uint8 conversion
is used only for PNG visualization, never for geometry or endpoint metrics.

Runs save atomic `summary.json` progress/completion state, loss histories,
reloadable model/optimizer states, initial noise, per-method native/decoded
float32 disk-backed trajectories, per-particle metrics, decoder control
metrics, endpoint/reconstruction/path PNG grids, and grid-refinement reports.
`comparison.csv` and `curvature_comparison.png` summarize both coupling runs
in every representation; the comparison is exploratory, not a reliable ranking.
`measurement_status` flags grid sensitivity separately from pipeline completion.
A completed run can still have unresolved curvature; inspect the refinement
report before interpreting differences between methods.
Code source hashes and peak process memory are recorded. Larger image
experiments can occupy multiple GB. Defaults reserve a 20 GB
payload/cache/run budget, use two CPU threads, and load one large representation
at a time; available disk/RAM are checked before execution. Failed/incomplete
runs and caches are reported explicitly and are not overwritten.
All default image tests use local synthetic fixtures and never download assets.
The reduced seed-42 pilot completed all ten models and both measurement spaces
with approximately 3 GiB peak process RAM. Its 8→16-step checks remain
grid-sensitive, so these outputs validate the pipeline rather than establish
a trustworthy representation ranking.

Useful flags include `--representations pixels`, `--seed`, `--train-steps`,
`--integration-steps`, `--batch-size`, `--hidden-width`, `--depth`, `--heads`,
`--n-train`, `--n-eval`, `--threads`, and `--disk-budget-gb`. Tiny training
budgets do not guarantee convergence. Use `--no-grid-refinement` only for an
explicitly labeled unrefined smoke test. The full five-space baseline is fixed
at 256×256; a pixel-only smoke test may use another size divisible by 16.
To scale up explicitly, use `--n-eval 16 --integration-steps 32`; frozen ViT-XL
decoding dominates CPU execution, especially with straight-path controls.

Released pretrained DiT inference is a separate future exploratory track.
A pretrained checkpoint cannot switch training coupling at inference, and
the RAE collection supplies neither the SD-VAE nor pixel-space DiT baseline.
Any such track requires another approved download manifest and documented
conditioning, prediction type, time schedule, ODE solver, and provenance.
It must not be mixed with the controlled ten-model comparison.

## Staged CIFAR-10 sweeps and results notebook

The read-only [image sweep notebook](notebooks/02_image_sweeps.ipynb) brings
the legacy pilot and new grid/training/capacity/seed results into one place.
Notebook execution **never trains models or downloads assets**. Set its
`RESULT_ROOT` (or `FLOW_SWEEP_RESULTS`) and `FILTERS` to inspect selected runs.

The separate sweep runner defaults to the approved CPU smoke profile:

| Control | Smoke settings |
| --- | --- |
| Representations / couplings | All five / independent and minibatch OT |
| CIFAR training / validation | 32 / 8 disjoint images from the training split |
| Generated native / decoded-path particles | 2 / 1 fixed particle |
| Training seeds | 42, 43 |
| Capacities (width, depth) | (32, 1), (64, 2) |
| Training checkpoints | Updates 10 and 25 from one continuous training stream |
| RK4 / measurement steps | 8, 16, 32 |

This is **40 training streams and 80 checkpoints**, not 80 retrainings.
The image split, validation RNG and generated-noise seed remain fixed across
training seeds. CIFAR's test split is reserved for final evaluation and is
not used for capacity/checkpoint selection. Raw validation loss is only
comparable within the same coupling objective.

```bash
# Inspect exact work counts and cumulative retained-output/cache disk usage.
python -m flow_curvature.image_sweeps --phase plan

# These are explicit compute commands, not automatic notebook operations.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  python -m flow_curvature.image_sweeps --phase all

# Repeating the same command resumes compatible incomplete work.
# Or separate training from checkpoint-only evaluation:
python -m flow_curvature.image_sweeps --phase train
python -m flow_curvature.image_sweeps --phase evaluate
python -m flow_curvature.image_sweeps --phase aggregate
```

Use the prepared project interpreter (`.venv/bin/python`) or activate its
environment. Optional dependencies are the same `image` and `notebook` extras.
This runner has no download flag: the previously approved local CIFAR/model
assets must already exist. It will refuse a projected cumulative 20 GB
cache/results allocation rather than delete existing outputs.

### Grid studies without retraining

Each checkpoint is integrated independently at every requested grid.
Solver metrics are evaluated on a **common coarse measurement grid** and
shared-time position errors are reported against the finest available solve.
Separately, nested subsamples of that finest solve test finite-difference
measurement resolution. The finest solve is a reference, not ground truth.

Only the selected finest-path particle is decoded, once per time point.
Coarser decoded measurement grids reuse those frames. Straight-native-path
decoder controls are retained. Pixel-space solver convergence is explicitly
**not** certified by these decoded measurement checks. Endpoint decoding
uses both generated particles for the validation quality estimate.
Metrics stay full-dimensional and continuous; PNG clipping is display-only.

`grid_checks` requires two successive passing refinements using
`absolute_tolerance + relative_tolerance * abs(refined_value)`.
Otherwise the status is `unresolved_at_max_grid`; the runner never silently
extends the approved workload. Near-zero and undefined metrics are explicit.
Passing these checks does not establish training convergence or remove the
input-projection bottleneck.

### Research configurations, selection and replication

`--config profile.json` accepts the documented `SweepConfig` fields. Start
from the `config` object printed by `--phase plan`, change only the controls
you intend to vary, and use a **new output directory** for a changed profile.
For example, research lists can use `checkpoints: [200, 1000, 5000]`,
`grids: [16, 32, 64, 128]` and `training_seeds: [42, 43, 44, 45, 46]`;
these are **not executed by the smoke defaults**. More demanding capacities,
including widths at least 768, are opt-in and must pass resource checks.
Do not treat small-width CPU smoke results as capacity-matched rankings.

After inspecting curves, an optional `--selection selection.json` explicitly
chooses stream IDs for later stages:

```json
{
  "schema_version": 1,
  "reason": "Example only: retain matched controls after checking quality and grid curves.",
  "stream_ids": [
    "dinov2-independent-w64-d2-s42",
    "dinov2-ot-w64-d2-s42"
  ]
}
```

Use IDs compatible with the supplied profile. Unknown/empty IDs fail rather
than silently launch a different run. Do not select solely by curvature.
Selection is for configurations, not evidence that omitted methods are worse.
Keep matched representation/coupling controls when defining GPU finalists.
Full research sweeps and five-seed runs require separate compute approval.

Snapshots save completed update, weights, optimizer, histories, independent
RNG streams, compatible data/transform identity and resumable batch hashes.
Interrupted training resumes from the latest complete snapshot; partial
evaluation is recomputed. A missing earlier checkpoint cannot be recovered
from later weights: the runner fails explicitly instead of relabeling them.
The old one-run image CLI stays supported.
Its legacy pilot appears separately in the notebook and is never pooled with
the new validation-split protocol.

`sweep.json` records per-stream status; each evaluation stores native, solver,
pixel and decoder-control metrics, image grids and validation quality.
`measurements.csv`, `aggregate.json` and `runtime.json` are produced by
aggregation. Seed summaries use one trained model as the replication unit;
particles are not independent training runs. Two smoke seeds show points and
ranges, not persuasive confidence intervals. At five or more seeds, reported
bootstrap intervals are explicitly exploratory and use model seeds.

### CPU and GPU runtime estimates

`runtime.json` reports observed setup, training/validation, OT assignment,
integration, decoding and diagnostic/I/O costs from this machine. Downloads
are excluded; cache creation versus reuse changes setup cost.
Process peak RSS belongs to each recorded execution, not the entire resumed
sweep. An evaluation-only cache check does not measure the original decoder
memory footprint; older execution records may lack this measurement.
It includes a clearly labeled CPU variability scenario, not a statistical
confidence interval. Rebenchmark new widths instead of assuming constant
cost per update.

No GPU is available on the current host. A100 tables are **unmeasured
sensitivity scenarios**, assuming 5x, 15x or 30x acceleration of accelerable
work while retaining measured CPU assignment and diagnostic/I/O costs.
They are not promised A100 speedups or GPU benchmarks; transfer overhead,
small batches and decoder behavior can dominate. Run the same profile on
the target device to replace these assumptions with measurements.

Keep CUDA dependencies separate from the existing CPU environment. On a
GPU host with a CUDA-13-compatible driver, create a separate environment and
install compatible CUDA wheels (verify driver support first):

```bash
uv venv --python 3.12 .venv-cuda
uv pip install --python .venv-cuda/bin/python \
  torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-cuda/bin/python -e '.[image,notebook]'
# In a separate profile JSON set "device": "cuda", then inspect before running:
.venv-cuda/bin/python -m flow_curvature.image_sweeps \
  --config gpu-profile.json --output-dir results/cifar-gpu --phase plan
```

These are GPU-host instructions, not a validated CUDA execution on this
CPU-only machine. Never reuse CPU trajectory caches as GPU measurements.
The CPU lock retains torchvision 0.29.2; the optional image requirement also
allows 0.29.1 because that compatible CUDA wheel is available on the verified
CUDA 13 index. GPU environments therefore have separately recorded versions.
Float32 model inference and float64 geometry remain the baseline;
requesting CUDA when unavailable fails explicitly. Do not enable mixed
precision or TF32 silently for curvature. The NumPy solver and SciPy OT
assignment stay on CPU, so CPU/GPU transfers are included in end-to-end costs.
No larger dataset or additional model download is part of this task.

## Background

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482)

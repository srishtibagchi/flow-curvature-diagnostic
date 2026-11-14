# Data

The runner deterministically generates independent Gaussian endpoint sets and
held-out source points in `processed/`. Binary data is gitignored; regenerate it
from the configuration and seed recorded in each run's `summary.json`.

Each `.npz` contains `source` and `target` arrays `(n_train, 2)` and `initial`
`(n_eval, 2)`. The filename encodes the seed and sample counts. The Gaussian
parameters are defined in `GaussianPair` and included in the run metadata.
No downloads or external datasets are required.

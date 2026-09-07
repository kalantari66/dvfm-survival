# Source and packaging notes

## HACSurv

The minimal `HACSurv_2D` port in `src/sota/hacsurv.py` was adapted with the
upstream repository owner's permission from
https://github.com/Raymvp/HACSurv at revision
`da945141058cf6ccd5cb175002d6bc35b6e9bd9a`. It retains the neural monotone
margins, stochastic mixture-of-exponentials generator, inverse autograd rule,
and bivariate observed-data likelihood, with local adaptations for explicit
device/dtype management, numerical checks, and validation-only checkpointing.

The implementation lives in `src/sota/hacsurv.py`.

## SOTA survival baselines

DeepHit, gradient-boosted survival analysis, random survival forest, and the
Weibull AFT wrapper were adapted from the author's MIT-licensed
`survival-copula` repository at revision
`0b4b031609185f510c892aea1269df90f0ac9a88`. The source used was under
`src/sota`, with training behavior checked against
`src/experiments/train_semisynthetic_datasets.py`. Imports and prediction
outputs were adapted to the unified DVFM runner; scientific model defaults are
recorded in the experiment YAML.

The two uploaded scripts are preserved byte-for-byte in `reference/`.

The supplied implementation is preserved in `reference/`. Its production
DVFM pieces were extracted without changing their formulas into
`src/dvfm/model.py`, `src/dvfm/training.py`, and `src/dvfm/prediction.py`.
Generic data generation and evaluation live under `src/utility`; competing
methods live under `src/sota`.

Packaging additions are limited to:

- YAML configuration and a command-line runner;
- generic CSV/XLSX loaders;
- real, synthetic, and semi-synthetic input interfaces;
- result-file organization;
- configuration validation that prevents accidental use of shortened DVFM settings;
- tests that check shapes, finite losses, prediction dimensions, metrics, and exact reference parameters.

The benchmark CLI has no `--quick` performance mode. Use `--validate-only` to check files and settings without model fitting.

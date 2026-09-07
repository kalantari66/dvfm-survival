# Source and packaging notes

## HACSurv

The minimal `HACSurv_2D` port in `src/dvfm/hacsurv.py` was adapted with the
upstream repository owner's permission from
https://github.com/Raymvp/HACSurv at revision
`da945141058cf6ccd5cb175002d6bc35b6e9bd9a`. It retains the neural monotone
margins, stochastic mixture-of-exponentials generator, inverse autograd rule,
and bivariate observed-data likelihood, with local adaptations for explicit
device/dtype management, numerical checks, and validation-only checkpointing.

The two uploaded scripts are preserved byte-for-byte in `reference/`.

The package uses `src/dvfm/reference_core.py`, copied from `reference/VAE_montcarlo.py`. One import-only compatibility change is applied there:

- use NumPy's `trapz` when `scipy.integrate.trapz` is unavailable in newer SciPy versions.

This does not change any model formula, loss, training step, prediction equation, generator, or metric calculation.

Packaging additions are limited to:

- YAML configuration and a command-line runner;
- generic CSV/XLSX loaders;
- real, synthetic, and semi-synthetic input interfaces;
- result-file organization;
- configuration validation that prevents accidental use of shortened DVFM settings;
- tests that check shapes, finite losses, prediction dimensions, metrics, and exact reference parameters.

The benchmark CLI has no `--quick` performance mode. Use `--validate-only` to check files and settings without model fitting.

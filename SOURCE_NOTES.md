# Source and packaging notes

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

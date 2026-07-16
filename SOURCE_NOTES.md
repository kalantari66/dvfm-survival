# Source and packaging notes

Algorithmic code was extracted from the two supplied Python files without changing the model equations or optimization procedures.

Packaging-only additions:

- module separation and imports
- YAML configuration and CLI
- generic CSV/XLSX loaders
- holdout and k-fold experiment wrappers
- output organization
- SciPy `trapezoid` compatibility alias for the former `trapz` name
- SurvivalEVAL 0.8 import fallback for `KaplanMeierArea`

The exact supplied files are retained in `reference/` for line-by-line comparison.

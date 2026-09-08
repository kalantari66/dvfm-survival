# SOTA model package

This directory is the single home for comparison-model implementations and
adapters. Missing baselines were ported from the MIT-licensed
`thecml/survival-copula` repository at revision
`0b4b031609185f510c892aea1269df90f0ac9a88`. HACSurv retains its separate
upstream provenance documented in the repository-level `SOURCE_NOTES.md`.

## Duplicate implementation decisions

| Model | Selected implementation | Comparison with `survival-copula` |
| --- | --- | --- |
| CoxPH | Existing DVFM/lifelines implementation | The external version uses scikit-survival plus automatic cleaning, scaling, column removal, and escalating ridge penalties. Those interventions would change the established baseline, so it was not substituted. |
| DeepSurv | `sota.baselines.train_deepsurv` | The external network has one 100-unit layer and validation patience; the retained benchmark has 64/32-unit layers with dropout and the established experiment interface. Both form Cox risk sets within configured batches. The external early-stopping code does not restore its best validation weights. |
| MTLR | `sota.baselines.train_mtlr` | The external version uses a linear MTLR coding matrix, explicit L2 penalty, and validation patience. We retain the benchmark's existing neural discrete-time formulation so completed and future comparisons remain consistent. |
| DeepHit | Ported external pycox implementation | Newly available through the unified adapter with validation-only early stopping. |
| GBSA / RSF | Ported external scikit-survival implementations | Defaults match the external experiment configuration; prediction curves are adapted to the common evaluation grid. |
| Weibull AFT | Ported external lifelines wrapper | Output is adapted to the common median and survival-curve contract. |
| Bayesian individual Cox--Gamma frailty | New implementation based on the [PyMC individual-frailty example](https://www.pymc.io/projects/examples/en/latest/survival_analysis/frailty_models.html) | Uses the same multiplicative Gamma subject frailty with a piecewise-exponential Cox baseline. The population parameters are MAP estimates; conjugacy gives each subject's exact conditional Gamma posterior, avoiding an infeasible 10,000-dimensional NUTS run. |
| HACSurv-2D | Existing authorized HACSurv port, moved here | Existing validation checkpointing, numerical checks, device handling, and joint-dependence diagnostics are preserved. |
| ClaytonAFT | Existing implementation, moved under `sota` | No equivalent model exists in the external `src/sota` folder. |

The external `utility` package was reviewed but is not copied wholesale. Its
data preprocessing and metrics overlap this repository's unified data, split,
and evaluation code, while its MTLR-specific loss is only required by the
unselected duplicate MTLR implementation.

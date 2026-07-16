# DVFM Survival

Clean, configuration-driven code for **Deep Variational Frailty Models (DVFM)** for survival prediction under dependent censoring.

The repository keeps the supplied model architecture, censored ELBO, training loop, aggregate-posterior Monte Carlo prediction, synthetic generators, and baseline algorithms unchanged. The new code only organizes them into modules and adds a generic interface for synthetic, real, and semi-synthetic datasets.

## Included methods

- DVFM with Weibull event/censoring decoders and a shared latent frailty
- Cox proportional hazards
- DeepSurv
- neural MTLR
- Clayton-Weibull AFT dependent-censoring baseline

## Metrics

- Concordance index (C-index)
- integrated Brier score with IPCW (IBS-IPCW)
- dependence-aware IBS (IBS-DEP)
- MAE on uncensored, censored, and all samples
- oracle MAE/IBS when true event times are available (synthetic and semi-synthetic data)

## Installation

```bash
git clone <YOUR-GITHUB-URL>
cd dvfm-survival
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## Quick smoke tests

```bash
dvfm-run --config configs/real_example.yaml --quick
dvfm-run --config configs/semi_synthetic_example.yaml --quick
dvfm-run --config configs/paper_synthetic.yaml --quick
```

Full paper-style synthetic benchmark:

```bash
dvfm-run --config configs/paper_synthetic.yaml
```

Full paper-style real-data benchmark:

1. Put the datasets in `data/raw/` using the paths/column names in `configs/paper_real.yaml`.
2. Run:

```bash
dvfm-run --config configs/paper_real.yaml
```

The real datasets are not redistributed in this repository. Check their original licenses and access requirements.

## Input formats

### Real data

Use CSV or Excel. Required columns are an observed follow-up time and an event indicator. All remaining columns are used as covariates unless `feature_cols` is given.

```yaml
mode: real
data:
  path: data/raw/my_data.csv
  time_col: time
  event_col: event
  feature_cols: [age, biomarker_1, treatment]
```

Event values may be numeric/Boolean or common labels such as `dead/alive`, `event/censored`, and `yes/no`.

### Semi-synthetic data

Provide real covariates together with simulated complete event and censoring times. The runner constructs `time = min(event_time, censor_time)` and `event = I(event_time <= censor_time)`.

```yaml
mode: semi_synthetic
data:
  path: data/raw/my_semi_synthetic.csv
  true_event_time_col: event_time
  true_censor_time_col: censor_time
```

### Synthetic data

Synthetic scenarios use the generator in the supplied code. Available mechanisms are `clayton`, `frank`, `gaussian`, `gumbel`, `frailty_mixture`, and `clayton_covdep`.

```yaml
mode: synthetic
synthetic:
  n_samples: 5000
  n_features: 10
  scenarios:
    - {id: S1, copula: clayton, theta: 1, dependence: low}
```

## Outputs

Each run writes:

- `results_raw.csv`: every repeat/fold
- `results_mean.csv`: grouped means
- `results_std.csv`: grouped standard deviations
- `resolved_config.json`: exact configuration used
- optional compressed prediction files when `save_predictions: true`

## Repository structure

```text
src/dvfm/model.py       DVFM architecture and C-ELBO
src/dvfm/training.py    DVFM training
src/dvfm/prediction.py  aggregate-posterior Monte Carlo prediction
src/dvfm/synthetic.py   synthetic dependent-censoring generators
src/dvfm/baselines.py   CoxPH, DeepSurv, MTLR, ClaytonAFT
src/dvfm/metrics.py     paper metrics
src/dvfm/runner.py      dataset-independent experiment pipeline
configs/                reproducible YAML configurations
reference/              original supplied scripts
paper/                  supplied manuscript
```

## Reproducibility note

The scenario parameters in `configs/paper_synthetic.yaml` reproduce Table 2 of the supplied manuscript. Their interpretation follows the parameterization in the supplied generator exactly; no copula or frailty formulas were redefined during packaging.

## Citation

Please cite the accompanying manuscript, *Deep Variational Frailty Models for Survival Prediction Under Dependent Censoring*.

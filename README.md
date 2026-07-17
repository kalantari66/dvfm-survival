# DVFM Survival — Reference-Parameter Repository

Clean GitHub repository for **Deep Variational Frailty Models (DVFM)** under dependent censoring.

This revision uses the supplied implementation directly and restores the active reference settings, especially **200 DVFM epochs**. The earlier five-epoch/sixty-epoch smoke configuration has been removed because it was not suitable for comparing model performance.

## What is unchanged

The following are imported from `src/dvfm/reference_core.py`, which is a package-compatible copy of the supplied `VAE_montcarlo.py`:

- DVFM encoder and decoder
- Weibull event/censoring likelihood
- censored ELBO
- KL annealing and optimizer loop
- aggregate-posterior Monte Carlo prediction
- synthetic dependent-censoring generators
- DeepSurv implementation
- MTLR implementation
- oracle and IPCW Brier-score functions

The only compatibility edit in the package copy is a fallback from SciPy's removed `trapz` import to NumPy's equivalent. The original uploaded scripts are preserved unchanged in `reference/`.

## Reference parameters

The main values are:

```text
DVFM epochs       200
DVFM learning rate 0.001
latent dimension   20
batch size          64
beta max             1.0
warm-up epochs      50
free bits             0
MC samples          100
DeepSurv epochs     200
MTLR epochs         200
MTLR bins           200
synthetic N       10000
features             10
test fraction       0.30
time-grid points    1000
```

See [`PARAMETER_AUDIT.md`](PARAMETER_AUDIT.md) for the complete mapping.

## Installation

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[test]"
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test]"
```

## Check the installation without changing parameters

```bash
dvfm-run --config configs/reference_original.yaml --validate-only
pytest -v
```

`--validate-only` checks the configuration and dataset paths. It does not fit models or generate performance numbers.

## Reproduce the active uploaded experiment

```bash
dvfm-run --config configs/reference_original.yaml
```

This configuration reproduces the active settings in `VAE_montcarlo.py`: Gaussian generator, theta 1, 10,000 samples, 70/30 split, 1,000 time points, and 200 epochs for DVFM/DeepSurv/MTLR.

## Full synthetic benchmark

```bash
dvfm-run --config configs/paper_synthetic.yaml
```

This is computationally expensive: 10 scenarios × 5 repetitions, with 200 training epochs and 100 Monte Carlo samples.

## Real datasets

Place the datasets in `data/raw/` with the names and columns listed in `configs/paper_real.yaml`, then run:

```bash
dvfm-run --config configs/paper_real.yaml
```

The repository does not redistribute restricted real datasets.

A generic real dataset can be CSV or Excel and must include:

- an observed follow-up-time column;
- an event indicator column (`1` event, `0` censored);
- feature columns.

## Semi-synthetic datasets

A semi-synthetic file contains real or user-provided covariates plus complete simulated event and censoring times. The loader constructs

```text
observed_time = min(event_time, censor_time)
event = 1(event_time <= censor_time)
```

Run the included format example with:

```bash
dvfm-run --config configs/semi_synthetic_example.yaml
```

## Outputs

Each experiment writes:

- `results_raw.csv`
- `results_mean.csv`
- `results_std.csv`
- `resolved_config.json`
- optional prediction arrays when enabled

## Repository structure

```text
src/dvfm/reference_core.py   package-compatible supplied algorithm code
src/dvfm/runner.py           real/synthetic/semi-synthetic experiment interface
src/dvfm/metrics.py          paper metrics
src/dvfm/baselines.py        CoxPH and ClaytonAFT wrappers; reference neural baselines
configs/                     exact reproducible parameter files
reference/                   unchanged uploaded scripts
paper/                       supplied manuscript
PARAMETER_AUDIT.md           parameter provenance
```

## Citation

Please cite the accompanying manuscript, *Deep Variational Frailty Models for Survival Prediction Under Dependent Censoring*.

# AGENTS.md

Repository structure, reproducibility protocol and development rules for autonomous coding agents
working in the DVFM survival repository. It covers how the repository is put together and how to
change it safely — not what the experiments found. For install and run commands see
[`README.md`](README.md); the experimental protocols are defined by the configurations in
[`configs/`](configs/) and the generators in [`src/utility/`](src/utility/).

## Structure

Research code for Deep Variational Frailty Models (DVFM) under dependent right censoring. A
research repository, not a library for general use.

The layers are:

- `src/dvfm/` — the model itself: encoder/decoder, censored ELBO, training loop, prediction.
- `src/sota/` — baselines behind a common adapter interface (`adapters.py`), including the
  vendored HACSurv-2D port.
- `src/utility/` — data-generating processes (`synthetic.py`, `semisynthetic.py`), metrics,
  splitting, runtime helpers.
- `src/experiments/` — configuration schema, runner, and the recovery diagnostics that turn a
  fitted model into rows of results.
- `configs/` — one YAML per study. The config is the experiment; code changes that alter a
  study's meaning belong in a new config, not a silent edit to an existing one.
- `workflows/` — [GWF](https://gwf.app/) submission targets, one per study. GWF is a Python
  workflow manager that expands a target script into SLURM jobs and reruns only what is
  missing. These are the cluster entry points.
- `scripts/` — per-dataset runners and the aggregation steps the workflows call.
- `notebooks/` — read completed artifacts from `results/` and write into `paper/`. They never fit
  a model.

The installed package exposes one console script, `dvfm-run` (`experiments.cli:main`). Studies are
selected by `--config`, never by editing defaults.

## Results are artifacts, not sources

`data/`, `results/` and `figures/` are git-ignored outputs. Never hand-edit a results CSV, never
commit one, and never fabricate a number to make a notebook run. If an artifact is missing, say it
is missing; a plot built from invented data is worse than no plot.

Notebooks consume artifacts and must fail loudly when a run is incomplete — the existing manifest
and row-count assertions are deliberate. Preserve them.

Figures and tables under `paper/` are generated. Edit the notebook cell that produces them, not
the output.

When reading `results_raw.csv`, filter to `is_primary_checkpoint == true` and
`prediction_mode == aggregate_posterior`; the other rows exist for ablation. Four further facts
about the artifacts will silently corrupt an analysis if ignored:

- `learned_conditional_kendall_tau` is written by both DVFM and HACSurv but computed differently,
  so only within-model comparisons are valid.
- `oracle_joint_survival_ise` is duplicated across `prediction_mode` rows, not recomputed per
  mode.
- Independence is the only scenario at `tau = 0`, and the copula set changes with tau, so code
  filtering by tau must account for it.
- DeepSurv output produced before commit `b74cebe` is invalid; the regression test is in
  `tests/test_sota.py`.

## Scientific integrity and safety

- Never invent a metric value, a run, a dataset row, or a citation.
- Report pre-existing failures separately from regressions you introduced.
- State uncertainty where it exists rather than resolving it silently; if two artifacts disagree,
  surface the disagreement.
- Assume everything committed is public. Never commit raw data, credentials, access-controlled
  extracts, or subject-level information. MIMIC-IV and SEER are access-controlled and their
  extracts stay out of the repository.
- Do not commit, push, open a pull request, submit a cluster job, or download data without
  explicit human authorization.

## Coordinated changes

A configuration field is one interface expressed in several places. Adding or renaming one must
update, together:

1. the schema and validation in `src/experiments/config.py`;
2. every `configs/*.yaml` that must keep working;
3. the consumer in `src/experiments/runner.py` or the relevant model;
4. the GWF target in `workflows/` if the field affects resources or target naming;
5. `tests/test_config.py`; and
6. the configuration-schema section of `README.md`, if the field is user-facing.

A change that alters the meaning of an existing study needs a new config and a new output
directory, so that old artifacts remain interpretable.

## Development rules

- Inspect the current branch, the working tree, the implementation, and nearby tests before
  writing.
- Make small, reviewable changes and preserve unrelated work.
- Match the conventions of the file you are editing; this repository is not uniform, and
  consistency within a file beats a global style.
- Add comments only for non-obvious reasons — why a clamp exists, why an extrapolation is linear —
  never to restate the code.
- Edit notebooks through `nbformat` or another structured parser and verify the result is valid
  JSON.
- Prefer a new test over a manual check when fixing a bug that produced wrong numbers.

Run the checks appropriate to the change, in the `dvfm` conda environment:

```bash
conda activate dvfm
pytest tests/                                          # full suite
pytest tests/test_config.py tests/test_sota.py         # targeted while iterating
dvfm-run --config configs/synthetic.yaml --validate-only
dvfm-run --config configs/semi_synthetic.yaml --validate-only
gwf -f workflows/semi_synthetic/workflow.py status     # before any submission
```

`--validate-only` resolves and checks a configuration without training and is the cheapest way to
confirm a config change is well formed.

## Pointers

| Topic | Source |
|---|---|
| Primary synthetic protocol | [`configs/synthetic.yaml`](configs/synthetic.yaml) |
| Semi-synthetic protocol | [`configs/semi_synthetic.yaml`](configs/semi_synthetic.yaml) |
| Data-generating processes and latent definitions | [`src/utility/synthetic.py`](src/utility/synthetic.py), [`src/utility/semisynthetic.py`](src/utility/semisynthetic.py) |
| Hyperparameter search spaces and selection | [`configs/semi_synthetic_tuning.yaml`](configs/semi_synthetic_tuning.yaml) |
| Metric definitions | [`src/utility/metrics.py`](src/utility/metrics.py) |
| Figure and table definitions | [`notebooks/`](notebooks/) |
| Environment and run commands | [`README.md`](README.md), [`environment.yml`](environment.yml) |


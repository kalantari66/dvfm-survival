# DVFM Survival

Research code for **Deep Variational Frailty Models (DVFM)** under dependent right censoring.

The repository asks one falsifiable question: under which structural conditions can a
shared-latent generative model recover the event-time distribution or the dependence-generating
frailty from right-censored observations? The study protocols are defined by the configurations in
[`configs/`](configs/), the data-generating processes by [`src/utility/`](src/utility/), and the
invariants any change must preserve by [`AGENTS.md`](AGENTS.md). The manuscript appendices give the
formal statement of each protocol.

---

## 1. Install

```bash
conda env create -f environment.yml
conda activate dvfm
```

`environment.yml` is the portable specification. `environment.lock.yml` is the exact resolved
environment that produced the published results, captured from the cluster with
`conda env export`; it is Linux/x86-64 specific and rebuilds in two steps, because the project
itself is installed from the working tree rather than from an index:

```bash
conda env create -n dvfm-lock -f environment.lock.yml
conda activate dvfm-lock
pip install -e . --no-deps
```

`--no-deps` is required: without it pip re-resolves this project's dependency ranges and can
upgrade packages the lock just pinned.

One environment contains [GWF](https://gwf.app/), the editable project, the test extras, and the
CUDA 13.0 PyTorch build. GWF is a Python workflow manager that turns a script of target
definitions into SLURM jobs, tracking which have already completed; this repository uses it to
submit each study to a cluster. Verify the environment with:

```bash
gwf --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
dvfm-run --help
```

If `conda` is not on `PATH` in Windows Command Prompt, activate it directly with
`CALL %USERPROFILE%\miniconda3\condabin\conda.bat activate dvfm`, or open an Anaconda Prompt.

---

## 2. Quick start

Everything runs through one console script. The study is chosen by `--config`; nothing is
selected by editing defaults.

```bash
# 1. Resolve and validate a configuration without training (seconds)
dvfm-run --config configs/synthetic.yaml --validate-only

# 2. A real study (hours; normally submitted through GWF -- see section 4)
dvfm-run --config configs/synthetic.yaml
```

`--validate-only` is the cheapest way to confirm a config change is well formed. Run it before
every submission.

---

## 3. The studies

Each study is one config, one output directory, and one GWF workflow. Everything in the paper
comes from the two rows in bold.

| Study | Config | Workflow | Output |
|---|---|---|---|
| **Primary synthetic** (DVFM only, `latent_dim` 0 vs 1) | `synthetic.yaml` | `synthetic` | `results/synthetic/` |
| **Semi-synthetic benchmark** (12 datasets, 8 models) | `semi_synthetic.yaml` | `semi_synthetic` | `results/semi-synthetic/` |
| Semi-synthetic hyperparameter tuning | `semi_synthetic_tuning.yaml` | `semi_synthetic_tuning` | `results/semi-synthetic-tuning/` |

Tuning runs first and writes the per-dataset hyperparameters that
`semi_synthetic.yaml` consumes under `models.tuned_by_dataset`.

---

## 4. Running on the cluster

Workflows are [GWF](https://gwf.app/) targets: each `workflow.py` declares the jobs a study needs
and their dependencies, and GWF submits only the ones that are not already done. Create the
environment once on the login node, then submit:

```bash
conda activate dvfm

gwf -f workflows/synthetic/workflow.py status
gwf -f workflows/synthetic/workflow.py run
```

Replace `synthetic` with any workflow name from the table above. Always check `status` before
`run`.

SLURM resources (cores, memory, walltime, partition, account) live in the `resources:` block of
the study's YAML, not in the workflow file. The workflow activates the same `dvfm` environment on
the compute node, and the configs set `compute.device: cuda` so a job fails loudly rather than
falling back silently to CPU.

The semi-synthetic benchmark is split finely: one GPU target per dataset × model × seed, followed
by CPU aggregation targets that assemble seed-, model-, and dataset-level summaries. Partial
progress is therefore resumable — rerunning `run` submits only what is missing.

---

## 5. What a run produces

Each study writes below its `study.output_dir`:

```text
results/<study>/
  _SUCCESS                  marker written only on a complete run
  run_manifest.csv          one row per fit, with status
  results_raw.csv           one row per fit × checkpoint × prediction mode
  results_mean.csv          aggregated across repeats
  results_std.csv
  resolved_config.yaml      the fully expanded configuration actually used
  dvfm_diagnostics.csv      latent diagnostics (DVFM studies)
  latent_recovery_test.csv  per-subject recovered vs true latent, where defined
```

`results/`, `data/` and `figures/` are git-ignored. They are outputs — never hand-edit them, and
never commit them.

When reading results, filter to `is_primary_checkpoint == true` and
`prediction_mode == aggregate_posterior`. Other rows exist for ablation and are not the reported
numbers.

---

## 6. Analysis and figures

Notebooks read completed artifacts and write into `paper/`. They never fit a model, and they fail
loudly if a run is incomplete.

| Notebook | Reads | Writes |
|---|---|---|
| `notebooks/synthetic_results.ipynb` | `results/synthetic/` | `paper/figures/synthetic_*.pdf` |
| `notebooks/semi_synthetic_results.ipynb` | `results/semi-synthetic/` | `paper/figures/semi_synthetic_*.pdf`, `paper/tables/*.tex` |
| `notebooks/semi_synthetic_event_distribution.ipynb` | `results/semi-synthetic/` | source vs generated distribution figures |

Run them from the repository root, or from `notebooks/` — both resolve the project root. Each
notebook is the specification of the figures and tables it writes: what a panel shows and how a
number is computed is defined by the cell that produces it, not by a separate document.

---

## 7. Configuration schema

Every runnable YAML uses the same top-level sections:

```yaml
schema_version: 1
workflow:       # optional GWF target settings
resources:      # SLURM cores, memory, walltime, partition, account
study:          # name, stage, output directory
seeds:          # repeat seeds; seed i drives generation, splitting and model init
compute:        # device and CPU threading
data:           # generator or file source, plus scenarios or a grid
split:          # holdout or cross-validation with a disjoint validation partition
preprocessing:  # train-fitted covariate transforms
models:         # enabled models and their settings
evaluation:     # time grid, primary metrics, saved artifacts
```

The loader supplies shared defaults, validates the schema, and expands `data.grid` into
deterministic atomic scenarios. New generators should expose Kendall's tau and a target censoring
rate; the preserved legacy copula generator still accepts its family-specific `theta` explicitly.

A change that alters the meaning of an existing study needs a **new** config and a new output
directory, so previously produced artifacts stay interpretable.

---

## 8. Tests

```bash
pytest tests/                                     # full suite
pytest tests/test_config.py                       # config schema and validation
pytest tests/test_sota.py                         # baselines, including regression tests
```

Report pre-existing failures separately from regressions.

---

## 9. Layout

```text
configs/          runnable study configurations
workflows/        one GWF submission target per study
scripts/          per-dataset runners and aggregation steps the workflows call
notebooks/        analysis of completed artifacts into paper/
src/dvfm/         DVFM model, training, prediction
src/experiments/  configuration schema, runner, recovery diagnostics
src/sota/         comparison models behind a common adapter interface
src/utility/      data-generating processes, metrics, splitting, runtime
tests/            configuration and functional tests
paper/            generated figures and tables consumed by the manuscript
```

Comparison models live under [`src/sota/`](src/sota/), behind a common adapter interface; a module
that was ported from an external implementation carries its provenance in its own header. The
primary synthetic benchmark is DVFM-only; comparison models are reserved for the semi-synthetic
study.

---

## 10. Further reading

| Topic | Source |
|---|---|
| Primary synthetic protocol | [`configs/synthetic.yaml`](configs/synthetic.yaml) |
| Semi-synthetic protocol | [`configs/semi_synthetic.yaml`](configs/semi_synthetic.yaml) |
| Data-generating processes and latent definitions | [`src/utility/synthetic.py`](src/utility/synthetic.py), [`src/utility/semisynthetic.py`](src/utility/semisynthetic.py) |
| Hyperparameter search spaces and selection | [`configs/semi_synthetic_tuning.yaml`](configs/semi_synthetic_tuning.yaml) |
| Metric definitions | [`src/utility/metrics.py`](src/utility/metrics.py) |
| Figure and table definitions | [`notebooks/`](notebooks/) |
| Rules for autonomous coding agents | [`AGENTS.md`](AGENTS.md) |

---

## Interpretation

Dependent censoring is not identifiable from right-censored observations without structural
assumptions. DVFM is evaluated as an inductive bias, not as a universal identification theorem or
as a test for informative censoring. Baseline prediction from `x`, retrospective frailty inference
from `(x, t, event)`, and population dependence recovery are separate tasks, and a method can
succeed at one while failing at another.

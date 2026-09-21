# DVFM Survival

Research code for **Deep Variational Frailty Models (DVFM)** under dependent right censoring.

The repository asks one falsifiable question: under which structural conditions can a
shared-latent generative model recover the event-time distribution or the dependence-generating
frailty from right-censored observations? See [`docs/SYNTHETIC.md`](docs/SYNTHETIC.md) for the
primary synthetic protocol, [`docs/DGP.md`](docs/DGP.md) for the data-generating processes, and
[`AGENTS.md`](AGENTS.md) for the invariants any change must preserve.

---

## 1. Install

```bash
conda env create -f environment.yml
conda activate dvfm
```

One environment contains GWF, the editable project, the test extras, and the CUDA 13.0 PyTorch
build. Verify it:

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

# 2. Two-epoch smoke test that exercises the full pipeline (minutes)
dvfm-run --config configs/frailty_recovery_smoke.yaml

# 3. A real study (hours; normally submitted through GWF -- see section 4)
dvfm-run --config configs/synthetic.yaml
```

`--validate-only` is the cheapest way to confirm a config change is well formed. Run it before
every submission.

---

## 3. The studies

Each study is one config, one output directory, and — for the studies that need a cluster — one
GWF workflow. Everything in the paper comes from the two rows in bold.

| Study | Config | Workflow | Output |
|---|---|---|---|
| **Primary synthetic** (DVFM only, `latent_dim` 0 vs 1) | `synthetic.yaml` | `synthetic` | `results/synthetic/` |
| **Semi-synthetic benchmark** (12 datasets, 8 models) | `semi_synthetic.yaml` | `semi_synthetic` | `results/semi-synthetic/` |
| Semi-synthetic hyperparameter tuning | `semi_synthetic_tuning.yaml` | `semi_synthetic_tuning` | `results/semi-synthetic-tuning/` |
| Frailty-recovery diagnostic | `frailty_recovery_diagnostic.yaml` | `frailty_recovery` | `results/frailty-recovery-diagnostic/` |
| HACSurv-2D feasibility | `hacsurv_synthetic_pilot.yaml` | `hacsurv_synthetic` | `results/hacsurv-gaussian-reference-pilot/` |
| Preserved reference run | `reference_original.yaml` | — | `results/reference-original/` |

Smoke configs run the same code paths in minutes and have no workflow:
`frailty_recovery_smoke.yaml` and `hacsurv_synthetic_smoke.yaml`.

---

## 4. Running on the cluster

Workflows are GWF targets. Create the environment once on the login node, then submit:

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
| `notebooks/frailty_recovery_diagnostic.ipynb` | `results/frailty-recovery-diagnostic/` | recovery diagnostics |

Run them from the repository root, or from `notebooks/` — both resolve the project root.
[`docs/FIGURES.md`](docs/FIGURES.md) specifies what each figure shows and how each number is
computed.

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
reference/        unchanged source implementation and provenance
docs/             manuscript and research notes
```

Comparison models live under [`src/sota/`](src/sota/); see
[`src/sota/README.md`](src/sota/README.md) for provenance and the comparison between overlapping
implementations. The primary synthetic benchmark is DVFM-only; comparison models are reserved for
the semi-synthetic study.

---

## 10. Further reading

| Topic | Source |
|---|---|
| Primary synthetic protocol | [`docs/SYNTHETIC.md`](docs/SYNTHETIC.md) |
| Data-generating processes and latent definitions | [`docs/DGP.md`](docs/DGP.md) |
| Hyperparameter-tuning protocol | [`docs/TUNING.md`](docs/TUNING.md) |
| Figure and table specifications | [`docs/FIGURES.md`](docs/FIGURES.md) |
| Vendored code and attribution | [`docs/SOURCE_NOTES.md`](docs/SOURCE_NOTES.md) |
| Reference-implementation parameter audit | [`docs/PARAMETER_AUDIT.md`](docs/PARAMETER_AUDIT.md) |
| Rules for autonomous coding agents | [`AGENTS.md`](AGENTS.md) |

---

## Interpretation

Dependent censoring is not identifiable from right-censored observations without structural
assumptions. DVFM is evaluated as an inductive bias, not as a universal identification theorem or
as a test for informative censoring. Baseline prediction from `x`, retrospective frailty inference
from `(x, t, event)`, and population dependence recovery are separate tasks, and a method can
succeed at one while failing at another.

# DVFM Survival

Research code for **Deep Variational Frailty Models (DVFM)** under dependent right censoring.

The repository asks one falsifiable question: under which structural conditions can a shared-latent generative model recover the event-time distribution or dependence-generating frailty from right-censored observations? See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the experimental design, metrics, ablations, and claim-adjustment rules.

## Install with Conda

```powershell
conda env create -f environment.yml
conda activate dvfm
```

If `conda` is not on `PATH` in Windows Command Prompt, activate it directly:

```bat
deactivate
CALL C:\Users\cml\miniconda3\condabin\conda.bat activate dvfm
```

The prompt should then begin with `(dvfm)`. Opening an **Anaconda Prompt** also makes `conda activate dvfm` available normally.

This single environment contains GWF, the editable project, tests, and the CUDA 13.0 PyTorch build. Verify it with:

```powershell
gwf --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Run

```powershell
# Validate without training
dvfm-run --config configs/reference_original.yaml --validate-only

# Preserved reference experiment
dvfm-run --config configs/reference_original.yaml

# Controlled size × tau × censoring × latent-dimension pilot
dvfm-run --config configs/synthetic_pilot.yaml

# Primary 10-seed synthetic benchmark (normally submit through GWF)
dvfm-run --config configs/synthetic.yaml

# Fast local check of the predictive hyperparameter machinery
dvfm-run --config configs/synthetic_hyperparameter_smoke.yaml

# Focused one-change decoder calibration ablation
dvfm-run --config configs/synthetic_dependence_calibration.yaml --validate-only

# Short HACSurv-2D integration check
dvfm-run --config configs/hacsurv_synthetic_smoke.yaml

# Two-epoch CUDA smoke test for the focused recovery diagnostic
dvfm-run --config configs/frailty_recovery_smoke.yaml

# Prespecified four-variant frailty-recovery diagnostic
dvfm-run --config configs/frailty_recovery_diagnostic.yaml

# File-based semi-synthetic smoke example
dvfm-run --config configs/semi_synthetic_example.yaml
```

Results are written below `results/<study-name>/` and include raw results, aggregated results, and the fully resolved configuration.

Submit the complete primary synthetic benchmark as one GWF target:

```bash
gwf -f workflows/synthetic/workflow.py status
gwf -f workflows/synthetic/workflow.py run
```

The complete hyperparameter screen is submitted as one GWF target:

```bash
gwf -f workflows/synthetic_hyperparameter/workflow.py status
gwf -f workflows/synthetic_hyperparameter/workflow.py run
```

The dependence-calibration ablation also runs as one target:

```bash
gwf -f workflows/dependence_calibration/workflow.py status
gwf -f workflows/dependence_calibration/workflow.py run
```

The focused HACSurv feasibility run is also one GWF target:

```bash
gwf -f workflows/hacsurv_synthetic/workflow.py run
```

## Canonical experiment specification

Every runnable YAML uses the same top-level sections:

```yaml
schema_version: 1
workflow:       # optional GWF target settings
resources:      # optional SLURM cores, memory, walltime, partition, account
study:          # identity, stage, output directory
seeds:          # separate DGP, sampling, split, and model seeds for new synthetic runs
compute:        # device and CPU threading
data:           # generator/file source and scenarios or grid
split:          # holdout or cross-validation with a disjoint validation partition
preprocessing:  # train-fitted covariate transforms
models:         # enabled models and settings
evaluation:     # time grid, primary metrics, saved artifacts
```

The loader supplies shared defaults, validates the schema, and expands `data.grid` into deterministic atomic scenarios. New generators should expose Kendall's tau and target censoring rate; the preserved legacy copula generator still accepts its family-specific `theta` explicitly.

Comparison models live under `src/sota/`. See
[`src/sota/README.md`](src/sota/README.md) for provenance and the comparison
between overlapping implementations. The primary synthetic benchmark includes
a scalable Bayesian individual Cox--Gamma frailty comparator with exact
per-subject conditional frailty posteriors.

## Run the synthetic pilot with GWF

Create and activate the project environment on the cluster login node:

```bash
conda env create -f environment.yml
conda activate dvfm
gwf --version
```

The synthetic YAML owns its SLURM resource settings. The complete grid runs in
one GPU target:

```bash
gwf -f workflows/synthetic/workflow.py status
gwf -f workflows/synthetic/workflow.py run
```

The focused frailty-recovery diagnostic has its own target:

```bash
gwf -f workflows/frailty_recovery/workflow.py status
gwf -f workflows/frailty_recovery/workflow.py run
```

The workflow activates the same `dvfm` environment on the compute node. Its experiment config sets `compute.device: cuda`, causing the job to fail instead of silently falling back to CPU.

## Layout

```text
configs/                 canonical runnable configurations
docs/                    manuscript and research notes
notebooks/               retained analysis and calibration notebooks
reference/               unchanged source implementation and provenance
src/dvfm/                DVFM model, training, and prediction only
src/experiments/         configuration and experiment orchestration
src/sota/                competing survival-model implementations
src/utility/             generic data, splitting, runtime, and metrics tools
tests/                   configuration and functional smoke tests
EXPERIMENTS.md            exact experimental and reporting design
```

## Interpretation

Dependent censoring is non-identifiable from right-censored observations without structural assumptions. DVFM is evaluated as an inductive bias, not as a universal identification theorem or a test for informative censoring. Baseline prediction from `x`, retrospective frailty inference from `(x,t,event)`, and population dependence recovery are separate tasks.

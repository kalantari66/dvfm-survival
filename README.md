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

# Small multi-mechanism pilot
dvfm-run --config configs/synthetic_pilot.yaml

# File-based semi-synthetic smoke example
dvfm-run --config configs/semi_synthetic_example.yaml
```

Outputs are written below `outputs/<study-name>/` and include raw results, aggregated results, and the fully resolved configuration.

## Canonical experiment specification

Every runnable YAML uses the same top-level sections:

```yaml
schema_version: 1
workflow:       # optional GWF target settings
resources:      # optional SLURM cores, memory, walltime, partition, account
study:          # identity, stage, output directory, seeds
compute:        # device and CPU threading
data:           # generator/file source and scenarios or grid
split:          # holdout or cross-validation
preprocessing:  # train-fitted feature/time transforms
models:         # enabled models and settings
evaluation:     # time grid, primary metrics, saved artifacts
```

The loader supplies shared defaults, validates the schema, and expands `data.grid` into deterministic atomic scenarios. New generators should expose Kendall's tau and target censoring rate; the preserved legacy copula generator still accepts its family-specific `theta` explicitly.

## Run the synthetic pilot with GWF

Create and activate the project environment on the cluster login node:

```bash
conda env create -f environment.yml
conda activate dvfm
gwf --version
```

The synthetic YAML owns its SLURM resource settings:

```bash
gwf -f workflows/synthetic/workflow.py status
gwf -f workflows/synthetic/workflow.py run
```

The workflow activates the same `dvfm` environment on the compute node. Its experiment config sets `compute.device: cuda`, causing the job to fail instead of silently falling back to CPU.

## Layout

```text
configs/                 canonical runnable configurations
docs/                    manuscript and research notes
notebooks/               retained analysis and calibration notebooks
reference/               unchanged source implementation and provenance
src/dvfm/                runner, models, data handling, metrics, baselines
tests/                   configuration and functional smoke tests
EXPERIMENTS.md            exact experimental and reporting design
```

## Interpretation

Dependent censoring is non-identifiable from right-censored observations without structural assumptions. DVFM is evaluated as an inductive bias, not as a universal identification theorem or a test for informative censoring. Baseline prediction from `x`, retrospective frailty inference from `(x,t,event)`, and population dependence recovery are separate tasks.

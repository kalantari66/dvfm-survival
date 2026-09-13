# Hyperparameter-tuning protocol

This document defines the development-only tuning protocol for the
semi-synthetic experiment suite.  It is deliberately separate from the final
10-seed evaluation runs, so that test performance is never used to select
hyperparameters.

## Development data and split

For each dataset, generate one semi-synthetic development cohort with:

```text
seed    = 99
copula  = Gaussian
Kendall tau = 0.50
```

Use the dataset's original censoring rate.  Split this cohort once into 70%
training and 30% validation, stratified by event status and discretized event
time.  There is no development test set: retaining a 20% test set would reduce
the data available to both fitting and selection without providing an unbiased
reported estimate.  The final experiment's held-out test splits, across its
10 seeds, remain untouched.

Tune using validation-set **oracle IBS** (`IBS Oracle`) only.  The
semi-synthetic DGP supplies the true event-time survival distribution, so
oracle IBS directly measures the primary estimand without needing a censoring
model.  Lower oracle IBS wins.

Record validation IPCW IBS (`IBS IPCW`) alongside every candidate, but use it
only as an observed-data-style sensitivity metric.  It assumes conditionally
independent censoring, which need not hold under the dependent-copula
conditions.  Do not use IPCW IBS to select hyperparameters or draw primary
semi-synthetic conclusions.  IBS-Dep is not computed or reported.

## Trials and outputs

Run 10 randomly sampled configurations for each tunable model family and each
dataset.  GWF creates one tuning job per dataset; each job evaluates every
model family.  Save:

- every sampled configuration, random-trial seed, validation objective, fit
  status, elapsed time, and stopping epoch;
- the selected configuration and its selection rule; and
- the fixed settings applied by the final semi-synthetic suite.

Tune DVFM with one latent dimension (`DVFM-z1`) and apply its selected shared
optimization/architecture settings to both DVFM-z1 and DVFM-z0.  The latent
dimension itself is an experimental condition, not a tuned parameter.

CoxPH has no tuning job unless an explicit benchmark regularization sweep is
introduced.  The `cox_penalizer` used to fit the semi-synthetic DGP is not a
CoxPH benchmark hyperparameter.

## Search spaces

Sample uniformly from the discrete values below.  The spaces are intentionally
compact: ten trials cannot reliably explore a large Cartesian grid.

| Model | Parameters |
| --- | --- |
| DeepSurv | learning rate `{1e-4, 3e-4, 1e-3, 3e-3}`; weight decay `{0, 1e-5, 1e-4, 1e-3}`; hidden layers `{[16], [32,16], [64], [64,32], [64,64,16]}`; dropout `{0.0, 0.1, 0.25}` |
| MTLR | learning rate `{3e-4, 1e-3, 3e-3, 1e-2}`; weight decay `{0, 1e-5, 1e-4, 1e-3}`; hidden layers `{[16], [32,16], [64], [64,32]}`; dropout `{0.0, 0.1, 0.25}`; time bins `{50, 100, 200}` |
| Random survival forest (RSF) | trees `{100, 300, 500}`; maximum depth `{3, 5, 8}`; minimum split samples `{2, 10, 25}`; minimum leaf samples `{1, 5, 10}`; maximum features `{sqrt, log2, 0.5}` |
| Gradient boosting survival analysis (GBSA) | estimators `{100, 300, 500}`; learning rate `{0.01, 0.05, 0.1}`; maximum depth `{1, 2, 3}`; minimum split samples `{2, 10, 25}`; minimum leaf samples `{1, 5, 10}`; maximum features `{sqrt, log2, 0.5}`; subsample `{0.6, 0.8, 1.0}` |
| Clayton AFT | learning rate `{1e-4, 5e-4, 1e-3, 5e-3, 1e-2}` |
| HACSurv | marginal learning rate `{3e-5, 1e-4, 3e-4}`; copula/marginal learning-rate multiplier `{0.3, 1, 3}`; hidden width `{16, 32, 64}`; scale regularization `{0.1, 1, 10}`; AdamW weight decay `{0, 1e-5, 1e-4, 1e-3}` |
| Cox--Gamma Frailty | learning rate `{0.003, 0.01, 0.03, 0.1}`; baseline intervals `{5, 10, 20}`; baseline smoothness `{0.1, 1, 10}`; coefficient-prior standard deviation `{1, 2.5, 5}` |
| DVFM-z1 | learning rate `{3e-4, 1e-3, 3e-3}`; weight decay `{0, 1e-5, 1e-4, 1e-3}`; encoder/decoder widths `{[32,16]/[16,32], [64,32]/[32,64], [64,64,16]/[16,64,64]}`; dropout `{0.0, 0.1, 0.25}` |

Keep fixed parameters fixed across trials, including the DVFM likelihood
configuration and the final-suite time grid (`n_time_points = 100`).

## Training and stopping policy

The stopping rule is part of the protocol, not a hyperparameter to optimize.
It must be identical in tuning and final runs for a given model.

- **DVFM:** retain the 200-epoch budget and select the post-warmup checkpoint
  with the best validation ELBO.  Do not add new early stopping solely for
  tuning.
- **HACSurv:** retain validation-likelihood early stopping.
- **Cox--Gamma Frailty:** retain validation marginal-NLL early stopping.
- **DeepSurv, MTLR, Clayton AFT:** retain fixed epoch budgets until
  validation-based stopping is implemented consistently for both tuning and
  final runs.

## Applying the result

After selecting one configuration per model family and dataset, freeze those
settings before launching the full semi-synthetic suite.  The final suite then
uses all configured copulas, tau conditions, and 10 evaluation seeds; it does
not re-tune per copula, tau, or evaluation seed.

Promote the resulting per-dataset winners to `models.tuned_by_dataset` in
`configs/semi_synthetic.yaml`.  The runner resolves that mapping at fit time,
so the GWF dataset job and its `--dataset` CLI invocation use the matching
settings.  The prior IPCW-selected mapping is retained as explicitly ignored
legacy audit data and must not be promoted.

The final aggregate writes `accuracy_primary_and_sensitivity.csv`.  Its
`Oracle IBS (primary; DGP ground truth)` column is the primary accuracy report;
its IPCW IBS column is labeled as the secondary sensitivity result and its
conditional-independent-censoring assumption.

This favors a controlled comparison over condition-specific optimum chasing.
If a model fails repeatedly for a selected setting, report the failures and
replace it only through a documented rerun of the development protocol.

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

Tune the final scalar-latent DVFM with one latent dimension (`DVFM-z1`). The
latent dimension is fixed rather than treated as a tuned parameter; the
semi-synthetic suite does not run the `DVFM-z0` ablation.

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

## Completed tuning results

The development sweep completed on 2026-09-13.  All 12 dataset jobs and the
cross-dataset aggregation have `_SUCCESS` markers.  The aggregate contains
960 planned trials (12 datasets x 8 model families x 10 trials): 958 succeeded
and 2 failed.  The failures were DVFM trial 4 on `flchain` (non-finite gradient
norm at epoch 161) and DVFM trial 7 on `employee` (non-finite gradient norm at
epoch 176).  Both dataset/model groups retained nine successful candidates,
and neither failed trial was selected.

The 96 recorded winners were independently recomputed from the aggregate trial
CSV.  Every winner is the successful row with minimum `selection_score` in its
dataset/model group, and `selection_score` equals validation oracle IBS for all
958 successful rows.  The selected parameters are frozen in
`models.tuned_by_dataset` in `configs/semi_synthetic.yaml`; DVFM's sampled
architecture is expanded to `encoder_hidden`/`decoder_hidden`, and HACSurv's
sampled multiplier is resolved to `copula_learning_rate`.  The complete trial
and winner records remain in `results/semi-synthetic-tuning/`.

### Selected validation oracle IBS (primary)

Lower is better.  Each cell is the score of that dataset/model family's
selected hyperparameter configuration; this is not a cross-model selection.

| Dataset | Frailty | Clayton AFT | DeepSurv | DVFM | GBSA | HACSurv | MTLR | RSF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whas | 0.146133 | 0.551325 | 0.175187 | 0.148096 | 0.151697 | 0.160635 | 0.139807 | 0.151836 |
| gbsg | 0.159329 | 0.581076 | 0.163983 | 0.161474 | 0.160599 | 0.176648 | 0.159386 | 0.162126 |
| metabric | 0.173255 | 0.370385 | 0.178150 | 0.174352 | 0.173914 | 0.171382 | 0.174874 | 0.176513 |
| churn | 0.138080 | 0.193027 | 0.150616 | 0.138493 | 0.140522 | 0.136404 | 0.146759 | 0.151542 |
| nacd | 0.150701 | 0.155895 | 0.161822 | 0.151614 | 0.151574 | 0.151771 | 0.157664 | 0.161241 |
| flchain | 0.079181 | 0.712729 | 0.102236 | 0.080097 | 0.079749 | 0.079847 | 0.079857 | 0.080743 |
| support | 0.217382 | 0.319760 | 0.222749 | 0.219355 | 0.220486 | 0.217811 | 0.223846 | 0.221461 |
| employee | 0.150714 | 0.197704 | 0.165016 | 0.130109 | 0.147982 | 0.124263 | 0.143631 | 0.151101 |
| mimic_iv | 0.177423 | 0.188300 | 0.149868 | 0.136664 | 0.138581 | 0.126864 | 0.195601 | 0.141541 |
| seer_brain | 0.170492 | 0.264921 | 0.191091 | 0.170872 | 0.170316 | 0.168614 | 0.172775 | 0.173197 |
| seer_liver | 0.186090 | 0.309629 | 0.206182 | 0.185127 | 0.187088 | 0.184316 | 0.188019 | 0.190165 |
| seer_stomach | 0.184502 | 0.344964 | 0.203980 | 0.184939 | 0.184774 | 0.183926 | 0.185958 | 0.187538 |

### Selected validation IPCW IBS (secondary sensitivity)

These values describe the oracle-selected configurations; they did not affect
selection and retain the conditional-independent-censoring caveat above.

| Dataset | Frailty | Clayton AFT | DeepSurv | DVFM | GBSA | HACSurv | MTLR | RSF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whas | 0.153176 | 0.546785 | 0.175656 | 0.155119 | 0.158228 | 0.167840 | 0.144577 | 0.157574 |
| gbsg | 0.147193 | 0.600851 | 0.152201 | 0.148260 | 0.149744 | 0.166417 | 0.146375 | 0.151096 |
| metabric | 0.171867 | 0.384831 | 0.176493 | 0.174037 | 0.173886 | 0.170361 | 0.175000 | 0.176663 |
| churn | 0.132051 | 0.204196 | 0.148131 | 0.140032 | 0.135455 | 0.136474 | 0.136975 | 0.142444 |
| nacd | 0.147542 | 0.157629 | 0.163586 | 0.151122 | 0.150749 | 0.150511 | 0.155605 | 0.161627 |
| flchain | 0.079007 | 0.723338 | 0.100799 | 0.079687 | 0.079518 | 0.079666 | 0.079809 | 0.080574 |
| support | 0.218676 | 0.321092 | 0.223771 | 0.220706 | 0.221624 | 0.219202 | 0.225146 | 0.222495 |
| employee | 0.111677 | 0.231968 | 0.148673 | 0.125644 | 0.119101 | 0.134350 | 0.116567 | 0.118692 |
| mimic_iv | 0.111752 | 0.233000 | 0.175494 | 0.173220 | 0.146872 | 0.150861 | 0.124266 | 0.151974 |
| seer_brain | 0.169347 | 0.267314 | 0.183072 | 0.169656 | 0.169533 | 0.168491 | 0.169948 | 0.171035 |
| seer_liver | 0.185608 | 0.330673 | 0.196166 | 0.186786 | 0.187348 | 0.186015 | 0.188993 | 0.189557 |
| seer_stomach | 0.179123 | 0.361770 | 0.195179 | 0.180993 | 0.179901 | 0.179838 | 0.181595 | 0.182442 |

## Applying the result

The selected configuration for each model family and dataset is frozen before
launching the full semi-synthetic suite.  The final suite then
uses all configured copulas, tau conditions, and 10 evaluation seeds; it does
not re-tune per copula, tau, or evaluation seed.

The per-dataset winners are stored in `models.tuned_by_dataset` in
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

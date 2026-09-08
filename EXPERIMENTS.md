# DVFM Experimental Design
## Scientific question
Under which structural conditions does a shared-latent generative model recover the event-time distribution and dependence-relevant frailty from right-censored observations?

DVFM is not presented as a general solution to non-identifiability. The experiments test whether its inductive bias is useful when shared latent heterogeneity is approximately correct, whether it is harmless under independence, and how it fails under misspecification.

## Claims and required evidence
1. **Event-distribution recovery:** DVFM improves oracle event-survival error under dependent censoring.
2. **Mechanism specificity:** the improvement comes from a latent path shared by the event and censoring models, rather than generic capacity.
3. **Frailty recovery:** when the DGP contains an explicit identifiable coordinate for shared frailty, posterior summaries recover its ordering and calibrated scale to a stated degree.
4. **Robustness:** benefits persist across realistic covariates and outcomes, dependence geometry, dependence strength, and censoring severity.
5. **Safe negative control:** at independence, DVFM does not invent material dependence or degrade event prediction substantially.

Claims must be weakened if the corresponding preregistered evidence below is absent.

## Tasks that must not be conflated
- **Baseline prediction:** estimate $$S_E(t \mid x)$$ for a new subject using only covariates.
- **Retrospective frailty inference:** estimate a subject's latent frailty from $$(x, \mathrm{observed\_time}, \mathrm{event})$$ after follow-up.
- **Population dependence recovery:** estimate the joint event/censoring mechanism or Kendall's tau.

Posterior frailty recovery after observing follow-up is not evidence of prospective individual prediction from `x` alone.

## Stage 0: engineering validation
Every generator must pass tests for deterministic seeding, finite positive times, target censoring tolerance (absolute error at most 0.02), target Kendall tau tolerance (absolute error at most 0.03 with a large calibration sample), train-only preprocessing, and absence of test-set model selection. Each atomic run writes its resolved config, seed, Git revision, runtime, achieved censoring rate, and empirical dependence.

## Stage 1: mechanistic synthetic benchmark (about 25% of paper evidence)
### Primary synthetic benchmark

`configs/synthetic.yaml` is the main paired benchmark. It uses 10 repeats of
10,000 subjects with 10 covariates over the same 12 shared-Gaussian-frailty
conditions as the pilot: `kendall_tau` in `{0, 0.25, 0.50, 0.75}` crossed with
censoring in `{0.25, 0.50, 0.75}`. This is a DVFM-only mechanistic experiment,
not a model-comparison benchmark.
Each seeded random holdout assigns 70% of subjects to training, 10% to
validation, and 20% to testing. The 120 paired datasets therefore produce
240 DVFM fits in one GWF target.

The primary DVFM uses `latent_dim = 1`, matching the one-dimensional true
frailty, and the same architecture with `latent_dim = 0` is an internal
conditional-independence control. Both use the recovery-validated 200-epoch schedule
(`beta_max = 1`, warmup 50, learning rate `0.001`, batch size 64), and the best
numerically valid validation-ELBO checkpoint at or after epoch 50, with Adam
weight decay `1e-4`. The full grid is submitted as one GWF target
with `workflows/synthetic/workflow.py` and writes to `results/synthetic/`.

The DGP draws ten independent standard-normal covariates and a standard-normal
subject frailty. Separate fixed Gaussian coefficient vectors govern event and
censoring times. The same calibrated frailty loading enters both Weibull AFT
predictors and is chosen to attain the requested conditional `kendall_tau`; a
censoring intercept is then calibrated to attain the requested censoring rate.
Observed follow-up is the minimum of latent event and censoring time. No
covariate or time scaling is applied.

### Current shared-Gaussian-frailty pilot

`configs/synthetic_pilot.yaml` runs five paired repeats over:

- Sample size: `10000`.
- `kendall_tau`: `{0.00, 0.25, 0.50, 0.75}`.
- Censoring rate: `{0.25, 0.50, 0.75}`.
- Fitted DVFM latent dimension: `{0, 1, 5}`.
- Models: `{CoxPH, DeepSurv, MTLR, ClaytonAFT, DVFM}`.

The DGP coefficients are fixed by `dgp_seed`; independent seed streams control sampling, splitting, and model optimization. Each sample-size/tau/censoring/repeat cohort is generated and censored once, then the identical train/validation/test subjects are reused by every model and latent dimension.

DVFM uses the recovery-validated default: 200 fixed epochs, `beta_max = 1`, warmup 50, and learning rate `0.001`. The primary checkpoint minimizes validation ELBO among numerically valid epochs 50--200, after beta has reached its final value. The final epoch is retained only as a secondary diagnostic when it is numerically valid. Reconstruction NLL is recorded but is never used for stopping or checkpoint selection. The legacy low-beta/reconstruction-checkpoint rule is not part of this pilot.

An epoch is numerically invalid if its training or validation ELBO, reconstruction NLL, or KL is non-finite or has absolute magnitude above the configured threshold of 100. Invalid epochs cannot supply the primary checkpoint. Non-finite training losses or gradients fail the fit immediately; invalid secondary final checkpoints are skipped and documented in `run_manifest.csv`.

### Predictive hyperparameter sweep

`configs/synthetic_hyperparameter_sweep.yaml` is a validation-driven screening
study around the frailty-recovery reference (`latent_dim = 1`, 200 epochs,
`beta_max = 1`, warmup 50, learning rate `0.001`, batch size 64, no dropout or
weight decay, decoder widths 32--64). It changes one factor at a time: dropout
0.10 and 0.25, 400 epochs, batch sizes 32 and 128, decoder widths 64--128, and
Adam weight decay `1e-4`.

The seven paired regimes cover Gaussian shared frailty under independence,
moderate and strong dependence, and censoring stress, plus moderate and strong
Clayton/Gamma shared frailty. The exact successful Gaussian cell at
`kendall_tau = 0.5` and 50% censoring is included. All variants see identical
subjects, censoring, and train/validation/test splits within a cell and seed.

Screening uses three seeds and one latent dimension, for 168 fits. The primary
comparison is prior-predictive oracle IBS on validation data; oracle CI is
secondary. Test results are written but never used to rank variants. A candidate
must also retain frailty Spearman correlation within 0.03 of the reference and
improve learned conditional Kendall's tau error. Confirm any selected candidate
with five fresh seeds before changing the paper default.

The sweep writes `hyperparameter_ranking.csv`,
`hyperparameter_scenario_summary.csv`, and `hyperparameter_paired_deltas.csv`,
in addition to the standard results,
training history, calibration curves, frailty diagnostics, and manifest. Run a
local smoke test with:

```bash
dvfm-run --config configs/synthetic_hyperparameter_smoke.yaml
```

Run the complete sweep as one GWF target with:

```bash
gwf -f workflows/synthetic_hyperparameter/workflow.py status
gwf -f workflows/synthetic_hyperparameter/workflow.py run
```

### Dependence-calibration decoder ablation

`configs/synthetic_dependence_calibration.yaml` targets the failure observed at
25% censoring: spurious dependence at `kendall_tau = 0` and underestimation at
`kendall_tau = 0.75`. It is a one-dimensional calibration curve over tau, not a
Cartesian grid. All runs use `n = 10000`, 10 features, latent dimension 1, five
fresh paired seeds, and the recovery-validated training settings with Adam
weight decay `1e-4`.

The decoder ablation is sequential so that each step changes one choice:

1. `reference`: conditional Weibull parameters with softplus scales and a
   nonlinear joint `(X,z)` decoder.
2. `exp_scale_link`: change only the scale link from softplus to exponential.
3. `additive_log_scale`: change only the latent pathway to
   `log(scale) = f(X) + a*z` for each margin.
4. `global_weibull_shape`: change only the two Weibull shapes from
   subject-specific outputs to globally learned parameters.

Compare adjacent steps, not only each treatment against the original
reference. Report learned conditional Kendall's tau and absolute error at every
target, oracle joint-survival ISE, subject-level frailty recovery, oracle IBS/CI,
checkpoint epoch, and numerical failures. No variant is selected using test
metrics.

Submit all four paired variants in one GWF target:

```bash
gwf -f workflows/dependence_calibration/workflow.py status
gwf -f workflows/dependence_calibration/workflow.py run
```

### HACSurv-2D feasibility baseline

HACSurv's bivariate single-event model is included as `hacsurv_2d` with
authorization from the upstream repository owner. It fits neural event and
censoring margins jointly with a learned mixture-of-exponentials Archimedean
copula. Because there are only two times, this is the non-hierarchical 2D member
of HACSurv rather than its competing-risks hierarchy.

`configs/hacsurv_synthetic_pilot.yaml` is deliberately a feasibility run: one
10,000-subject Gaussian shared-frailty cohort at `kendall_tau = 0.5` and 50%
censoring, using the same DGP, split, and model seed as the DVFM pilot. It runs
only HACSurv-2D. The model checkpoint is selected by validation likelihood only
after copula optimization begins. Evaluation reports marginal event-survival
oracle IBS, CI, MAE, calibration, learned Kendall's tau, training history,
runtime, and numerical failures. HACSurv does not produce subject-level frailty.

Run the quick CUDA check and the one-target cluster pilot with:

```bash
dvfm-run --config configs/hacsurv_synthetic_smoke.yaml
gwf -f workflows/hacsurv_synthetic/workflow.py run
```

At `kendall_tau = 0`, true-frailty correlation is undefined as a recovery target; those cells evaluate learned dependence, latent collapse, and predictive safety under independence. For positive tau, the best latent coordinate, its sign, and its affine calibration are selected using validation subjects only before held-out test recovery is calculated.

The result table contains oracle IBS, oracle concordance index, oracle MAE, and
`oracle_joint_survival_ise`. The latter is the subject-averaged, area-normalized
two-dimensional integrated squared error between the learned and true
conditional joint survival $P(T>t,C>c\mid X)$ on held-out synthetic subjects.
For DVFM this integrates over the prior latent distribution; for HACSurv it
evaluates the learned survival copula on its learned margins. Separate compact
artifacts store ELBO/reconstruction/KL trajectories,
final-versus-post-warmup-ELBO checkpoint diagnostics, active latent dimensions,
learned conditional `kendall_tau`, frailty recovery by censoring subgroup, and
population calibration curves. Full per-subject survival NPZ files are not
produced.

All evaluation implementations are centralized in `src/utility/metrics.py`.
SurvivalEVAL supplies curve interpolation, Harrell and Uno/IPCW concordance,
IPCW IBS, censoring-aware margin MAE, and the corresponding fully observed
oracle calculations. IBS-DEP is not computed.

### Focused frailty-recovery training diagnostic

Before extending the grid, `configs/frailty_recovery_diagnostic.yaml` freezes one condition: `kendall_tau = 0.50`, 50% censoring, fitted latent dimension 1, 8,873 subjects, and five independent sampling/split/model seeds. It runs both the shared-Gaussian DGP and a Marshall--Olkin Clayton construction whose standardized log-Gamma frailty is observed by the evaluator. The Clayton construction reproduces the latent mechanism used by the earlier successful SUPPORT experiment, but uses synthetic Weibull margins; it is therefore a mechanism replication, not a literal reproduction of the SUPPORT cohort.

The four DVFM variants isolate checkpointing and optimization:

| Variant | `beta_max` | Warmup | Learning rate | LR scheduler | Primary checkpoint |
| --- | ---: | ---: | ---: | --- | --- |
| `current` | 1.0 | 50 | 0.001 | validation ELBO | final |
| `checkpoint_fix` | 1.0 | 50 | 0.001 | validation ELBO | best validation reconstruction NLL |
| `old_training` | 0.2 | 150 | 0.0005 | validation reconstruction NLL | best validation reconstruction NLL |
| `schedule_isolation` | 1.0 | 150 | 0.0005 | validation reconstruction NLL | best validation reconstruction NLL |

Both final and best-reconstruction checkpoints are retained for every variant. Checkpoint selection, sign alignment, and affine latent calibration use validation data only. The untouched test partition supplies frailty Pearson/Spearman, validation-calibrated RMSE/R-squared, and oracle prediction metrics. An oracle-Z decoder determines whether the decoder and DGP can recover event-time behavior when the true subject frailty is supplied.

Every epoch records training and validation reconstruction NLL, ELBO, KL, validation frailty Pearson/Spearman, active latent dimensions, learned conditional `kendall_tau`, beta, and learning rate. Compact CSV/GZIP artifacts replace full prediction NPZ files:

- `training_history.csv.gz`: epoch trajectories.
- `frailty_recovery.csv`: checkpoint-level recovery, alignment, and subgroup metrics.
- `subject_latent_diagnostics.csv.gz`: held-out posterior means/standard deviations and calibrated frailty estimates.
- `results_raw.csv`: prior, aggregate-posterior, and oracle-Z prediction metrics.
- `calibration_curves.csv.gz`: population survival calibration summaries.
- `run_manifest.csv`: successes, failures, and runtimes.
- `checkpoints/`: final and best-reconstruction states.

Run the local CUDA smoke test first, then submit the diagnostic:

```bash
dvfm-run --config configs/frailty_recovery_smoke.yaml
gwf -f workflows/frailty_recovery/workflow.py status
gwf -f workflows/frailty_recovery/workflow.py run
```

### Data-generating mechanisms
1. Gaussian shared frailty with known `z_shared`.
2. Clayton copula (lower-tail dependence).
3. Gumbel copula (upper-tail dependence).
4. Frank copula (symmetric, no tail dependence).
5. Shared plus event-private and censor-private frailties with all three latents known.
6. Misspecified dependence: a mixture/sign-changing or direct event-to-censoring mechanism not representable by one shared frailty.

Copula experiments have known event/censoring quantiles and joint law, but not a uniquely recoverable Gaussian subject frailty. They must not report proxy coordinates as true frailty recovery.

### Confirmatory grid
- Kendall tau: `{0.00, 0.25, 0.50, 0.75}`.
- Target censoring rate: `{0.25, 0.50, 0.75}`.
- Fitted shared latent dimension: `{0, 1, 5, 10, 20}`.
- True shared dimension for explicit-frailty DGPs: `{1, 5}`.
- Sample size: `{1000, 5000, 10000}` for the Gaussian mechanism; `10000` for the full mechanism comparison.
- Seeds: `0..9`.
- Covariates: 10, with fixed DGP coefficients across model comparisons for a seed.

The full Cartesian product is not run blindly. Pilot runs identify invalid or computationally redundant cells; exclusions are documented before confirmatory runs.

### Models and ablations
- CoxPH.
- DeepSurv.
- MTLR.
- Correctly specified copula model for each copula DGP when available.
- Misspecified Clayton copula model.
- Joint Weibull model without a latent.
- Event-only latent model.
- Censor-only latent model.
- Separate event/censor latents without a shared path.
- Original DVFM shared latent.
- DVFM with censoring likelihood removed.
- Oracle-Z decoder for explicit-frailty DGPs only.

All neural comparisons use matched optimization budgets and comparable decoder capacity. Hyperparameters are selected using validation observed-data likelihood or a prespecified training rule, never oracle test metrics.

## Stage 2: semi-synthetic benchmark (about 75% of paper evidence)
Use 5--8 datasets spanning sample size, feature dimension, and nonlinear signal. Candidate survival or positive-regression sources include SUPPORT, METABRIC, GBSG, WHAS, FLCHAIN, STEEL, and AIRFOIL, subject to licensing and a documented preprocessing sheet.

Two generators are required:
1. **Foomani Algorithm 4:** retain real covariates and complete positive outcomes; fit the event marginal on the training partition only; map outcomes to event quantiles; sample censoring quantiles conditionally using a chosen copula; invert a calibrated censoring marginal; then form $$(\min(E,C), \mathbf{1}[E \le C])$$.
2. **Explicit shared frailty:** retain real covariates, generate event and censoring times from realistic fitted baselines plus known shared/private latent effects, and store every latent and complete time.

Confirmatory semi-synthetic grid:
- Mechanism: `{Gaussian frailty, Clayton, Frank, Gumbel}`.
- Kendall tau: `{0.00, 0.50, 0.75}`.
- Target censoring: `{0.25, 0.50, 0.75}`.
- Fitted latent dimension: `{0, 1, 5, 20}`.
- Seeds: `0..9`.

Use the same train/validation/test subjects for every method within a scenario. Generator fitting and censoring calibration use training data only. Complete event times remain hidden from model fitting and are used only for oracle evaluation.

## Outcomes
### Primary
- Oracle integrated Brier score over a prespecified common time interval.
- Integrated squared survival error against the known conditional event survival curve when available.
- Oracle event-time MAE, overall and separately for eventually censored/uncensored subjects.

### Mechanistic
- Absolute error in marginal and conditional Kendall tau.
- Joint-distribution error when the DGP density or survival function is available.
- Pearson/Spearman correlation between inferred and true latent coordinates after validation-only sign/permutation alignment.
- Validation-calibrated latent RMSE and R-squared.
- Posterior interval coverage and width.
- Per-dimension KL, active dimensions, and decoder sensitivity to the DVFM latent.

### Secondary
- Harrell C-index, IPCW IBS, and dependence-aware estimators. These are not primary evidence under dependent censoring because their estimators can themselves be biased or model-dependent.

## Statistical reporting
Report every scenario, not only an aggregate across mechanisms. Use paired seed-level differences against each comparator with 95% bootstrap confidence intervals. Report medians and interquartile ranges in addition to means when failures or heavy tails occur. Failed runs remain in the run manifest and are not silently discarded. Multiple-comparison-adjusted p-values are optional; effect sizes and uncertainty are mandatory.

## Decision rules for paper claims
- Claim robust event recovery only if DVFM improves the primary oracle metric over the strongest independence baseline in a clear majority of dependent-censoring cells and across most semi-synthetic datasets.
- Claim shared-frailty recovery only where explicit ground-truth frailty exists and recovery holds on held-out subjects across seeds.
- Claim robustness to copula misspecification only if DVFM competes with correctly specified copula baselines and beats a misspecified copula baseline across multiple families.
- Claim safety under independence only if the performance loss is practically negligible and learned dependence/active shared capacity remains near zero.
- If improvements occur only for aligned Gaussian frailty, position DVFM as a specialized shared-frailty model and remove broad dependence claims.

## Canonical configuration schema
All runnable configs use these top-level sections:
```yaml
schema_version: 1
workflow:       # optional GWF target settings
resources:      # optional SLURM cores, memory, walltime, partition, account
study:          # name, stage, output_dir
seeds:          # separate dgp, sampling, split, and model seeds for new synthetic runs
compute:        # device, torch_num_threads
data:           # source, dimensions/path, scenarios or grid
split:          # strategy, test_fraction/folds
preprocessing:  # train-fitted covariate preprocessing, such as zscore_x
models:         # enabled models and model-specific settings
evaluation:     # time grid, stored artifacts, primary metrics
```
`data.scenarios` lists explicit atomic conditions. `data.grid` is a Cartesian shorthand. Dependence is recorded canonically as Kendall's tau in new generators; family-specific parameters are derived and stored separately. The legacy `synthetic_copula` generator still requires `theta` explicitly until calibrated generators replace it.

## Execution stages
1. `smoke`: tiny data, one seed, interface correctness only.
2. `pilot`: small scenario subset used to diagnose training and set fixed hyperparameters.
3. `confirmatory`: frozen configuration and seeds; no selection on test oracle outcomes.
4. `sensitivity`: clearly labeled variations performed after confirmatory analysis.

Run or validate any canonical configuration with:
```bash
dvfm-run --config configs/<name>.yaml --validate-only
dvfm-run --config configs/<name>.yaml
```
The canonical synthetic configuration can also be submitted as one GWF target:
```bash
conda env create -f environment.yml
conda activate dvfm
gwf -f workflows/synthetic/workflow.py status
gwf -f workflows/synthetic/workflow.py run
```

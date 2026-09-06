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

- **Baseline prediction:** estimate `S_E(t | x)` for a new subject using only covariates.
- **Retrospective frailty inference:** estimate a subject's latent frailty from `(x, observed_time, event)` after follow-up.
- **Population dependence recovery:** estimate the joint event/censoring mechanism or Kendall's tau.

Posterior frailty recovery after observing follow-up is not evidence of prospective individual prediction from `x` alone.

## Stage 0: engineering validation

Every generator must pass tests for deterministic seeding, finite positive times, target censoring tolerance (absolute error at most 0.02), target Kendall tau tolerance (absolute error at most 0.03 with a large calibration sample), train-only preprocessing, and absence of test-set model selection. Each atomic run writes its resolved config, seed, Git revision, runtime, achieved censoring rate, and empirical dependence.

## Stage 1: mechanistic synthetic benchmark (about 25% of paper evidence)

### Current shared-Gaussian-frailty pilot

`configs/synthetic_pilot.yaml` runs the first controlled pilot with 1,000
subjects per generated cohort: 5 paired repeats
over `kendall_tau = {0, 0.25, 0.50, 0.75}`, `censoring_rate = {0.25, 0.50,
0.75}`, and fitted `latent_dim = {0, 1, 5}`. The DGP coefficients are fixed by
`dgp_seed`; independent seed streams control sampling, splitting, and model
optimization. Each scenario/repeat is generated and censored once, then the
identical train/validation/test cohort is reused for every latent dimension.

The synthetic result table contains only oracle IBS, oracle concordance index,
and oracle MAE (overall and by observed censoring status). Separate artifacts
store training reconstruction/KL trajectories, active latent dimensions,
learned conditional `kendall_tau`, population calibration curves, prior versus
aggregate-posterior predictions, posterior parameters, and the held-out true
frailty `z`.

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

1. **Foomani Algorithm 4:** retain real covariates and complete positive outcomes; fit the event marginal on the training partition only; map outcomes to event quantiles; sample censoring quantiles conditionally using a chosen copula; invert a calibrated censoring marginal; then form `(min(E,C), I[E<=C])`.
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
preprocessing:  # standardize_x, time_normalization
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

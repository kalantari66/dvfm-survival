# Results

Reference for the two evaluation studies behind the paper: what each figure and
table shows, and exactly how each number is computed. Written so the definitions
and aggregation rules can be lifted into the paper's captions and methods text.

Both notebooks read completed artifacts from `results/` and write into
`paper/figures/` and `paper/tables/`. Neither fits a model; rerunning a notebook
regenerates figures from whatever is currently on disk.

| | Synthetic | Semi-synthetic |
|---|---|---|
| Notebook | `notebooks/synthetic_results.ipynb` | `notebooks/semi_synthetic_results.ipynb` |
| Config | `configs/synthetic.yaml` | `configs/semi_synthetic.yaml` |
| Results | `results/synthetic/` | `results/semi-synthetic/` |
| Question | Does the shared latent do what it claims? | Does DVFM compete against baselines? |
| Models | DVFM only (`latent_dim` 0 vs 1) | 8 models |

---

## 1. Synthetic study

### Design

Fully simulated data from a Gaussian shared-frailty DGP (`data.source:
gaussian_shared_frailty`), so the true event times, true censoring times and the
true latent `z` are all known.

- 10,000 subjects, 10 covariates
- Grid: Kendall's tau in {0, 0.25, 0.50, 0.75} x censoring rate in {0.25, 0.50, 0.75}
- 10 repeats; each repeat independently reseeds generation, splitting and fitting
- Two model variants per cell: `latent_dim = 0` (ablation, no shared latent) and
  `latent_dim = 1` (the primary model)
- 4 x 3 x 10 x 2 = **240 fits**; the notebook asserts this count and refuses to
  plot if any fit failed or if the artifact sizes disagree

Only rows with `is_primary_checkpoint == true` and `prediction_mode == prior`
enter the figure. The primary checkpoint is
`best_validation_elbo_post_warmup`, so the reported model is the one selected on
validation ELBO after the KL warm-up, not the last epoch.

### Figure: `paper/figures/synthetic_main.pdf`

Four panels, each plotting a quantity against target Kendall's tau, with one
line per censoring rate (25% / 50% / 75%). Shaded bands are **mean +/- 1 SD
across the 10 repeats**.

**Panel A — Individual frailty recovery.** Spearman correlation between DVFM's
recovered per-subject latent and the true `z`, for `latent_dim = 1`, restricted
to `subgroup == "all"`. **tau = 0 is omitted**: with no shared-frailty signal the
latent is unidentified, so recovery is undefined rather than zero.

Recovery strengthens monotonically with dependence, and degrades under heavy
censoring only at weak dependence:

| tau \ censoring | 25% | 50% | 75% |
|---|---|---|---|
| 0.25 | 0.586 | 0.556 | 0.423 |
| 0.50 | 0.810 | 0.808 | 0.764 |
| 0.75 | 0.947 | 0.944 | 0.935 |

**Panel B — Joint-survival gain from the latent.** Paired difference
`ISE(z=0) - ISE(z=1)` in oracle joint-survival ISE, computed *within each
repeat* before averaging, so the two variants are compared on identical data.
Positive favours the shared-latent model. The dashed line marks zero; the axis
is scaled by 1e-3.

The metric is identical to the semi-synthetic one (Figure 2b below), and the
config values match: for each held-out subject the model predicts the bivariate
surface S(t, c | x) = P(T > t, C > c | x) on a 15 x 15 grid of time pairs
against truth from 2,000 DGP draws, for 128 subjects. Squared error is
integrated over both axes and **divided by the grid area**, then averaged over
subjects, so it is a mean squared error in probability units and `sqrt(ISE)`
reads as an RMS error in the joint survival probability.

**Panel C — Dependence calibration.** Estimated conditional Kendall's tau
against its target, for `latent_dim = 1`. The dashed diagonal is perfect
calibration; below it is underestimation, above it overestimation. This panel
plots tau, **not** latent-`z` values.

**Panel D — Event-prediction gain from the latent.** Paired
`oracle_ibs(z=0) - oracle_ibs(z=1)`, same pairing as Panel B. Positive favours
the shared-latent model. Axis scaled by 1e-2.

Panels B and D share a structure worth stating in the caption: **both are paired
differences within a repeat**, which removes cohort-to-cohort variation and
makes the band a paired SD, not a between-variant SD.

#### The tau = 0 penalty

At tau = 0 both gains are negative: the shared latent makes the model **worse**,
and this is a real effect rather than noise. It should be stated explicitly in
the paper — a reader who notices the negative region and finds it unmentioned
will discount the rest.

Panel B, joint-survival ISE at tau = 0 (paired, 10 repeats):

| censoring | ISE z=0 | ISE z=1 | gain | relative | t |
|---|---|---|---|---|---|
| 25% | 0.00287 | 0.00623 | -0.00336 | -117% | -4.2 |
| 50% | 0.00203 | 0.00603 | -0.00400 | -198% | -5.8 |
| 75% | 0.00245 | 0.00711 | -0.00466 | -190% | -11.5 |

Panel D, oracle IBS at tau = 0:

| censoring | IBS z=0 | IBS z=1 | gain | relative |
|---|---|---|---|---|
| 25% | 0.0961 | 0.1004 | -0.0043 | -4% |
| 50% | 0.0966 | 0.1185 | -0.0218 | -23% |
| 75% | 0.1021 | 0.1670 | -0.0650 | **-64%** |

**Why.** At tau = 0 the DGP has no shared frailty, so the `z = 0` ablation is
correctly specified while `z = 1` carries a latent the data cannot identify. Its
posterior stays near the prior, and marginalising over an uninformative latent
spreads predictive mass and induces dependence that is not there. The penalty
grows with censoring because heavier censoring leaves less event information to
pin `z` down — at 75% censoring the latent is least constrained and does the
most damage.

**Scale caveat for Panel B.** This is a large *relative* penalty on a small
*absolute* error: `sqrt(ISE)` is about 7.8 versus 5.0 probability points of RMS
error on the surface, so both variants recover the joint distribution well. The
1e-3 axis makes the negative region look comparable in size to the positive
gains at tau = 0.75, but those are ~95% relative improvements against this ~190%
relative degradation.

**Crossover.** Both panels cross zero at roughly tau = 0.2-0.3, and the benefit
then grows with both dependence and censoring — at tau = 0.75 with 75%
censoring, +69% on IBS and +96% on joint ISE (absolute IBS gain 0.194).

**Suggested framing.** Present this as a property of the evaluation rather than
something to explain away: the shared latent is not free capacity. It costs when
there is no dependence to capture and earns its place when there is, with a
clean crossover in between. That is a more credible claim than a model that
never loses.

---

## 2. Semi-synthetic study

### Design

Real covariates, simulated outcomes. For each source dataset a Cox model is
fitted to the real data to give the marginals; event and censoring times are
then drawn through a copula, so true event times and true censoring times are
known while the covariate structure stays realistic.

- **12 datasets**: whas, gbsg, metabric, churn, nacd, flchain, support,
  employee, mimic_iv, seer_brain, seer_liver, seer_stomach
- **5 scenarios**: independence at tau = 0, plus Gaussian / Clayton / Frank /
  Gumbel at tau = 0.5
- **10 seeds**; every model sees the same cohort and split within a seed
- **8 models**: CoxPH, DeepSurv, MTLR, RSF, ClaytonAFT, HACSurv,
  BayesianCoxGammaFrailty, DVFM
- 70/10/20 split, covariates z-scored on training statistics
- 12 x 5 x 8 x 10 = **4,800 rows** in `results_raw.csv`

Per-dataset hyperparameters live in `models.tuned_by_dataset`, selected by a
separate random search (10 trials per model per dataset) on a development cohort
with seed 99, scored by oracle IBS. That development cohort's validation split
is never reused by the reporting runs.

### Metrics

All primary metrics are **oracle** metrics: scored against the DGP's true event
times, not against the censored observations. This is the point of the
semi-synthetic design and should be stated explicitly in the paper, since it is
what separates these numbers from ordinary IPCW-style estimates.

| Metric | Definition |
|---|---|
| `oracle_ibs` | Integrated Brier score of the predicted survival curve against true event times. **Primary selection metric.** |
| `oracle_ci` | Concordance between true event times and the predicted median survival time |
| `oracle_mae` | Mean absolute error between true event time and predicted median survival time |
| `oracle_joint_survival_ise` | See Figure 2b below |
| `frailty_spearman` | Spearman correlation between DVFM's recovered latent and the true `z` |
| `absolute_conditional_kendall_tau_error` | \|learned tau - target tau\| |
| `ibs_ipcw` | Observed-data sensitivity metric only; assumes conditionally independent censoring and is **not** used for any selection or conclusion |

`oracle_ci` and `oracle_mae` are not computed from risk scores. Both go through
a predicted **median survival time**, defined here as the first point on the
shared evaluation grid where S(t) <= 0.5, **falling back to the last grid
point** when the curve never crosses. The evaluation grid is 100 uniform points
from 0 to the 95th percentile of training event times, shared by all models.

**Deviation from SurvivalEVAL.** IBS comes from SurvivalEVAL and the
concordance calculation is theirs, but the predicted times fed into it are ours,
not the package's `predict_median_st`. SurvivalEVAL differs in two ways: it
linearly interpolates between grid points to find the exact crossing, and where
a curve never reaches 0.5 it extrapolates a line anchored at (0, 1) rather than
returning a constant. Ours therefore quantises predictions to the grid (90-99
distinct values instead of one per subject) and ties every non-crossing subject
at the same value.

Measured impact is small. Scoring Cox curves both ways on the semi-synthetic
cohorts:

| | fallback rate | `oracle_ci` ours -> SurvivalEVAL | `oracle_mae` ours -> SurvivalEVAL |
|---|---|---|---|
| nacd (37% censoring) | 33% | 0.6957 -> 0.7068 | 26.4 -> 27.5 |
| mimic_iv (67% censoring) | 8% | 0.6413 -> 0.6413 | 731.2 -> 770.2 |

Note the fallback rate tracks how far the grid extends relative to curve decay,
not the censoring rate directly. The reported results retain our definition; the
appendix should state it, since the package is cited. Not tested: whether
per-model differences in fallback rate shift the relative ranking. On `mimic_iv`
Cox curves cross for 91.8% of subjects and DeepSurv's for 78.3%, so the tie
penalty is not identical across models, though the measured magnitude above
makes a rank change unlikely.

### Figure 1 — Model ranks (`semi_synthetic_model_ranks.pdf`)

Three panels (Oracle CI, Oracle IBS, Oracle MAE), one row per model, at
**tau = 0.5 only**.

Computation, in order:

1. **Rank within a single seed.** Inside each
   (dataset, copula, tau, censoring, repeat) cell, the 8 models are ranked
   against each other. Models are therefore compared on identical cohorts and
   splits.
2. **Average the 10 seeds** within each (dataset, copula, model).
3. **Average the 4 copulas** within each (dataset, model), giving one mean rank
   per dataset per model.

Because the grid is balanced at exactly 10 repeats per cell, steps 2 and 3
commute — averaging copulas first gives identical numbers. The notebook raises
`Incomplete seed grid` if that balance is violated.

4. **The plotted marker is the median of those 12 per-dataset mean ranks.**
5. **The error bar is a 95% bootstrap confidence interval** for that median:
   resample the 12 datasets with replacement 10,000 times, take the median each
   time, report the 2.5% and 97.5% quantiles.

> **Two things the caption must say.** The interval is a bootstrap CI for a
> median over **datasets**, not a standard deviation and not a spread over
> seeds — seed variation is deliberately averaged away in step 2, since seeds
> are paired across models while datasets are the population being generalised
> over. And **the vertical order is not a performance ranking**: DVFM is pinned
> to the bottom row by construction, with the baselines above it ordered by mean
> rank.

Current median ranks (lower is better; regenerate after any rerun):

| Model | Oracle CI | Oracle IBS | Oracle MAE |
|---|---|---|---|
| Cox-Gamma Frailty | 3.88 | 2.79 | 3.32 |
| CoxPH | 4.56 | 2.89 | 3.24 |
| HACSurv | 2.59 | 3.19 | 3.30 |
| DVFM | 2.85 | 3.22 | 3.06 |
| DeepSurv | 5.61 | 4.66 | 4.54 |
| ClaytonAFT | 2.98 | 5.61 | 5.16 |
| RSF | 6.39 | 5.81 | 6.22 |
| MTLR | 6.49 | 7.71 | 6.61 |

### Figure 2 — Recovery diagnostics (`semi_synthetic_recovery.pdf`)

**DVFM only.** Two panels, grouped by copula. Aggregation is
`dataset_balanced_summary`: average the 10 seeds within each
(dataset, copula, tau), then report **mean +/- 1 SD across the 12 datasets**.
Each bar is therefore 12 dataset-level values.

**Panel A — Individual shared-latent recovery.** Spearman correlation between
recovered and true `z`. Independence is excluded (undefined at tau = 0), so
this panel has 4 bars while Panel B has 5.

```
clayton 0.750   frank 0.754   gaussian 0.746   gumbel 0.727     SD 0.05-0.08
```

**Panel B — Dependence calibration.** \|learned tau - target tau\|, lower better.

```
independence 0.174   clayton 0.250   frank 0.275   gaussian 0.289   gumbel 0.314
```

The SDs here (about 0.16 on means of about 0.28) are large relative to the
differences between families, so the between-copula ordering in this panel is
not well separated and should not be claimed as a finding.

> **Note the inconsistency with Figure 1**, which uses a median plus bootstrap
> CI. Figure 2 uses mean +/- 1 SD. Both are defensible but they are different
> quantities, and a reader will assume consistency unless each caption says
> which it is.

### Figure 2b — Joint-distribution recovery, DVFM vs. HACSurv

`semi_synthetic_joint_recovery_dvfm_vs_hacsurv.pdf`.

`oracle_joint_survival_ise` is the only recovery diagnostic that **both** DVFM
and HACSurv export; it is empty for all six other baselines. The latent-recovery
and tau-calibration metrics are DVFM-only, so no head-to-head is possible for
Figure 2's panels.

**The metric.** For each held-out subject the model predicts a bivariate
surface S(t, c | x) = P(T > t, C > c | x) on a 15 x 15 grid of time pairs
(0 to the 95th percentile of training times on each axis), for 128 subjects,
with the truth obtained from 2,000 DGP draws. The squared error is integrated
over both axes by trapezoid, **divided by the grid area**, and averaged over
subjects. Area normalisation makes it a mean squared error in probability
units, so `sqrt(ISE)` reads directly as an RMS error in the joint survival
probability.

It is strict because the surface's edges *are* the marginals — S(t, 0 | x) is
T's own survival curve and S(0, c | x) is C's — while the interior is the
coupling. One number can only be small if both marginals and their dependence
are right at every grid point. Kendall's tau collapses all of that to a single
rank-association scalar; empirically the two are uncorrelated here (within-model
Spearman between \|tau error\| and ISE is -0.10 for DVFM and +0.06 for HACSurv,
both p > 0.4), which is the argument for reporting this separately from
Figure 2's Panel B rather than as a refinement of it.

**Panels.** Left: grouped bars by copula, dataset-balanced mean +/- 1 SD across
datasets. Right: per-dataset paired dots on a log axis, pooled over copulas.
The left panel's whiskers are dominated by cross-dataset scale spread (ISE
ranges roughly 60x across datasets) rather than by any difference between the
models, so the right panel carries the model comparison.

**Caveats for the paper.**

- The two models are at **parity**, and the aggregate mean is carried almost
  entirely by one dataset. Excluding `flchain`, DVFM 0.0167 vs HACSurv 0.0181,
  paired Wilcoxon p = 0.57. With it, 0.0389 vs 0.0469, p = 0.12.
- **`flchain` is a shared failure**: RMS error above 0.5 in probability for both
  models, nearly identical across all five copulas including independence. That
  invariance means it is not a dependence-modelling failure, and it coincides
  with that dataset having the *best* oracle IBS of the twelve. Treat it as an
  open diagnostic, not a result.
- Which model wins is a **dataset property, not a copula property** — 9 of 12
  datasets have all five copulas agreeing on the winner. If the difference were
  about recovering dependence structure it should vary with copula family.
- Both models are shared-frailty constructions (DVFM marginalises a Gaussian
  latent by Monte Carlo; HACSurv's Archimedean generator phi(t) = E[exp(-Mt)] is
  the Laplace transform of a latent frailty), so parity is unsurprising. The
  real asymmetry is that DVFM infers a **per-subject** latent through an encoder
  while HACSurv marginalises a population-level one, and that DVFM's dependence
  can vary with covariates while HACSurv's generator is global.
- The semi-synthetic DGP applies a **single scalar copula parameter to every
  subject**, so its dependence is covariate-homogeneous — exactly HACSurv's
  assumption. This benchmark cannot exercise DVFM's covariate-varying dependence.

### Figure 3 — Subject-level latent recovery

Hexbin and scatter views of recovered vs. true `z` for a single dataset,
pooling held-out subjects across repeats, split by censoring status and by
copula family. Each repeat keeps its own validation-derived affine calibration;
no test subjects are used to fit that mapping. These answer robustness rather
than headline performance: recovery does not degrade for censored subjects, and
holds across copula families.

### Tables

- **`semi_synthetic_datasets.tex`** — source dataset characteristics: N, raw and
  encoded feature counts, original censoring rate, split.
- **`semi_synthetic_model_summary_tau0.tex`** and
  **`semi_synthetic_model_summary_tau0p5.tex`** — one table per dependence
  level; tau levels are never pooled. Each entry is the **median across the 12
  datasets of a model's mean seed-level rank**, i.e. exactly the quantity
  Figure 1 plots, so a table number can be read straight off the corresponding
  figure panel. The W/T/L column is a paired oracle-IBS record against DVFM,
  computed within that tau level, with a 1e-4 tie tolerance and reported from
  each comparator's perspective. Labels are
  `tab:semi_synthetic_models_tau0` and `tab:semi_synthetic_models_tau0p5`.

---

## 3. Known issues to carry into the paper

- **DeepSurv results before commit `b74cebe` are invalid.** Its Cox partial
  likelihood multiplied an `(N,)` event mask by an `(N, 1)` risk-score column,
  which broadcast to `(N, N)`; the sum factorised so that the event indicator
  cancelled, training the model as if nothing were censored. Damage scaled with
  censoring rate (Spearman -0.71 against the deficit versus CoxPH) and inverted
  the risk ordering entirely on `mimic_iv`. Fixed, with a regression test in
  `tests/test_sota.py`; all DeepSurv numbers were retuned and rerun afterwards.
- **Do not compare Kendall's tau across DVFM and HACSurv.** Both write to the
  `learned_conditional_kendall_tau` column but use different estimators: DVFM
  takes an empirical tau from 2,000 sampled (T, C) pairs at the mean covariate
  vector, HACSurv evaluates the analytic generator integral. One is a noisy
  single-point sample estimate, the other exact. Within-model comparisons are
  fine.
- **`oracle_joint_survival_ise` exists only at tau = 0 and tau = 0.5**, so there
  is no tau sweep for that metric in the semi-synthetic study.
- The **independence scenario is the only one at tau = 0**; the runner emits one
  product-copula condition rather than four duplicates. Any code filtering by
  tau must account for the copula set changing with tau.

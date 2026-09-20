# Results

Reference for the two evaluation studies: what each figure shows, how each
number is computed, and which claims the numbers support. Kept short on purpose
— ICLR gives 8 pages, so this is the source for captions and a paragraph or
two, not a results section in itself.

Both notebooks read completed artifacts from `results/` and write into
`paper/figures/` and `paper/tables/`. Neither fits a model.

| | Synthetic | Semi-synthetic |
|---|---|---|
| Notebook | `notebooks/synthetic_results.ipynb` | `notebooks/semi_synthetic_results.ipynb` |
| Config | `configs/synthetic.yaml` | `configs/semi_synthetic.yaml` |
| Results | `results/synthetic/` | `results/semi-synthetic/` |
| Question | Does the shared latent do what it claims? | Does DVFM compete against baselines? |
| Models | DVFM only (`latent_dim` 0 vs 1) | 8 models |

**Model class.** DVFM does not assume proportional hazards: the decoder emits a
per-subject Weibull shape *and* scale from `(x, z)`, so hazard ratios vary with
time. Only the `shape_mode: global` ablation reduces to PH. The semi-synthetic
DGP margins *are* CoxPH fits, so there the data satisfies PH while the model
does not assume it.

---

## 1. Synthetic study

10,000 subjects, 10 covariates, Gaussian shared-frailty DGP, so true event
times, true censoring times and the true latent `z` are all known. Grid:
tau in {0, 0.25, 0.50, 0.75} x censoring in {0.25, 0.50, 0.75}, 10 repeats,
`latent_dim` 0 vs 1 = **240 fits**. Only `is_primary_checkpoint == true` and
`prediction_mode == aggregate_posterior` rows enter the figures; the primary
checkpoint is `best_validation_elbo_post_warmup`.

Four standalone figures, one line per censoring rate, bands are **mean +/- 1 SD
across the 10 repeats**:

| File | Content |
|---|---|
| `synthetic_frailty_recovery.pdf` | Spearman(recovered `z`, true `z`); tau = 0 omitted (latent unidentified, so recovery is undefined, not zero) |
| `synthetic_joint_survival_gain.pdf` | Paired `ISE(z=0) - ISE(z=1)`, oracle joint survival |
| `synthetic_dependence_calibration.pdf` | Learned conditional tau vs target; dashed diagonal is perfect calibration |
| `synthetic_event_prediction_gain.pdf` | Paired `oracle_ibs(z=0) - oracle_ibs(z=1)` |

Both gain figures are **paired within a repeat** before averaging, so the band
is a paired SD, not a between-variant SD. Say this in the caption.

**Recovery** strengthens monotonically with dependence and degrades under heavy
censoring only at weak dependence:

| tau \ censoring | 25% | 50% | 75% |
|---|---|---|---|
| 0.25 | 0.586 | 0.556 | 0.423 |
| 0.50 | 0.810 | 0.808 | 0.764 |
| 0.75 | 0.947 | 0.944 | 0.935 |

**Calibration** at N = 10,000 is good in the middle and compressed at both ends:
learned tau is −0.146 (SD 0.367) at target 0, 0.248 at 0.25, **0.440 at 0.50**,
0.516 at 0.75. The target-0 estimate is unstable even here.

**The tau = 0 penalty is real and should be stated, not hidden.** With no shared
frailty the `z = 0` ablation is correctly specified, while `z = 1` carries a
latent the data cannot identify; marginalising over it spreads predictive mass
and induces dependence that is not there. Worst case (75% censoring): oracle IBS
−64%, joint ISE −190%. The penalty grows with censoring because fewer events
leave `z` less constrained. Scale caveat: that is a large relative penalty on a
small absolute error (`sqrt(ISE)` 7.8 vs 5.0 probability points).

Both gains cross zero at **tau ≈ 0.2–0.3** and grow with dependence and
censoring: +69% IBS and +96% joint ISE at tau = 0.75 with 75% censoring.

> **Framing.** The shared latent is not free capacity — it costs when there is
> no dependence and earns its place when there is, with a clean crossover. That
> is more credible than a model that never loses.

---

## 2. Semi-synthetic study

Real covariates, simulated outcomes: a Cox model per dataset gives the
marginals, a copula couples event and censoring times.

- **12 datasets** x **5 scenarios** (independence at tau = 0; Gaussian /
  Clayton / Frank / Gumbel at tau = 0.5) x **8 models** x **10 seeds** =
  **4,800 rows**
- 70/10/20 split, covariates z-scored on training statistics
- Per-dataset hyperparameters from a separate random search (10 trials, seed 99)
  on a development cohort whose validation split the reporting runs never reuse

All primary metrics are **oracle** metrics, scored against the DGP's true event
times rather than censored observations. State this explicitly — it is what
separates these numbers from IPCW-style estimates. `ibs_ipcw` is a sensitivity
check only and drives no conclusion.

`oracle_ci` and `oracle_mae` go through a predicted median survival time from
SurvivalEVAL's `predict_median_st`, which interpolates the S(t) = 0.5 crossing
and extrapolates from (0, 1) when a curve never reaches it. Extrapolated medians
can exceed the evaluation grid (on `mimic_iv`, 10,793 vs a grid max of 2,450),
inflating `oracle_mae`; an identically-1 curve yields `inf`, which the rank
machinery treats as a failure ranked below the worst successful model.

### Figure 1 — Model ranks (`semi_synthetic_model_ranks.pdf`)

Three panels (Oracle CI / IBS / MAE) at **tau = 0.5 only**. Rank the 8 models
within each (dataset, copula, tau, censoring, repeat) cell, average the 10
seeds, then the 4 copulas, giving one mean rank per dataset per model. The
marker is the **median of those 12 values**; the bar is a **95% bootstrap CI**
for that median over 10,000 resamples of the 12 datasets.

| Model | Oracle CI | Oracle IBS | Oracle MAE |
|---|---|---|---|
| CoxPH | 4.56 | 2.89 | 3.24 |
| RSF | 6.39 | 5.81 | 6.22 |
| MTLR | 6.49 | 7.71 | 6.61 |
| DeepSurv | 5.61 | 4.66 | 4.54 |
| ClaytonAFT | 2.98 | 5.61 | 5.16 |
| HACSurv | 2.59 | 3.19 | 3.30 |
| Cox-Gamma Frailty | 3.88 | 2.79 | 3.32 |
| DVFM | 2.85 | 3.22 | 3.06 |

> **Caption must say** that the interval is a bootstrap CI for a median over
> **datasets** (seed variation is averaged away in step 2, since seeds are
> paired across models while datasets are the population generalised over), and
> that **row order is fixed for readability, not a performance ordering**.

No model reaches rank 1 because the marker is a median of means: a model would
have to win nearly every seed and copula in at least half the datasets. Rank 1
does occur at the level where ranking happens — for Oracle CI, DVFM takes it in
159 of 480 paired seed cells, ahead of ClaytonAFT (125) and HACSurv (117).

### Figure 2 — Recovery diagnostics (DVFM only)

Two figures, both 7.5 x 6.0 in for a subfigure pair.

**`semi_synthetic_latent_recovery.pdf`** — Spearman(recovered `z`, true `z`),
mean +/- 1 SD across the 12 datasets, one bar per dependent copula
(independence excluded, undefined at tau = 0):

```
clayton 0.750   frank 0.754   gaussian 0.746   gumbel 0.727     SD 0.05-0.08
```

**`semi_synthetic_dependence_calibration.pdf`** — |learned tau − target tau| as
a 12 x 5 heatmap, every dataset x copula cell, seeds averaged. This replaced a
copula-averaged bar chart, because the pooled view hid the four findings below.

### The calibration result, stated honestly

1. **Copula is the least informative axis.** Median within-dataset SD across the
   four dependent copulas is 0.023; between-dataset SD is 0.165 — a ~7x
   difference. Do not claim a between-copula ordering.
2. **The error is one-directional.** All 48 dataset x copula cells at tau = 0.5
   *under*estimate. Mean learned tau is 0.21 against a target of 0.50.
3. **Two regimes, not a spectrum.** SUPPORT 0.04 and SEER (brain) 0.07 against
   METABRIC 0.53, WHAS 0.49, GBSG 0.47, FLChain 0.43 — the last four have
   learned tau ≈ 0, i.e. the model reports independent margins on data generated
   with tau = 0.5.
4. **Learned tau is largely a dataset-level constant.** Across datasets,
   learned tau at target 0 and at target 0.5 correlate at **0.95**, and a 0.5
   change in true tau moves the estimate by only **0.072** (slope ≈ 0.14).
   SUPPORT reports 0.34 when the truth is 0 and 0.47 when it is 0.5; WHAS
   reports ~0.01 in both. This is why the Independence column of the heatmap is
   *inverted* relative to the copula columns, and it means SUPPORT's good score
   is partly luck rather than calibration.

**Recovery and calibration are not in conflict.** `frailty_spearman` is
rank-based and monotone-invariant — it measures whether the *ordering* of `z` is
recovered. The learned tau is simulated from the decoder at the mean covariate
and depends on the *magnitude* of `z`'s effect on both margins. An attenuated
latent scale leaves Spearman at 0.75 while driving tau to 0. Expect a reviewer
to ask; the one-line answer is "ordering yes, scale no".

### Figure 2c — Identifiability (`semi_synthetic_tau_vs_training_events.pdf`)

Learned tau against uncensored training events (log axis), one marker per
dataset per target, dashed lines at both targets. Spearman rho between learned
tau at target 0.5 and event count is **+0.85**: WHAS (155 events) and GBSG (212)
sit on the tau = 0 line; SUPPORT (4,221) and SEER (brain) (4,188) reach 0.47 and
0.44. The vertical stub is the response to the true dependence, and it is short
everywhere.

Two exceptions to a pure sample-size reading: METABRIC has 772 events and
*negative* learned tau, and FLChain has 1,511 events (72.5% censored) and sits
with the small datasets.

> **The defensible claim**: given enough uncensored events the latent scale is
> identified and the learned tau approaches its target; below roughly 1,000
> events it collapses toward independence. The fully synthetic run supports this
> (0.440 at target 0.5, N = 10,000), but note the confound — that DGP's Weibull
> margins are well specified for the decoder, while the semi-synthetic margins
> are CoxPH fits.

### Figure 2d — Joint recovery, DVFM vs HACSurv

`semi_synthetic_joint_recovery_by_copula.pdf` (grouped bars by copula) and
`semi_synthetic_joint_recovery_by_dataset.pdf` (paired per dataset, log axis),
same size for a subfigure pair.

`oracle_joint_survival_ise` is the only recovery diagnostic both models export.
For each held-out subject the model predicts S(t, c | x) on a 15 x 15 grid
(128 subjects, truth from 2,000 DGP draws); squared error is integrated over
both axes and **divided by the grid area**, so `sqrt(ISE)` reads as an RMS error
in the joint survival probability. The surface's edges are the marginals and its
interior the coupling, so one number is small only if both are right —
empirically it is uncorrelated with |tau error| (within-model Spearman −0.10 for
DVFM, +0.06 for HACSurv), which is why it is reported separately.

Caveats, all of which belong in the text:

- **Parity.** 0.0389 vs 0.0469, paired Wilcoxon p = 0.12; DVFM lower in 37 of
  60 cells. Excluding `flchain`: 0.0167 vs 0.0181, p = 0.57.
- **`flchain` is a shared failure** — RMS error above 0.5 in probability for
  both models, nearly identical across all five copulas including independence,
  so it is not a dependence-modelling failure. Open diagnostic, not a result.
- Which model wins is a **dataset property, not a copula property**: 9 of 12
  datasets have all five copulas agreeing on the winner.
- Both are shared-frailty constructions, so parity is unsurprising. The
  asymmetry is that DVFM infers a **per-subject** latent and can vary dependence
  with covariates; HACSurv's generator is global. **The DGP applies one scalar
  copula parameter to every subject**, which is exactly HACSurv's assumption, so
  this benchmark cannot exercise that advantage.

### Figure 3 and tables

`semi_synthetic_subject_frailty_recovery.pdf` plus the hexbin views: recovered
vs true `z` for one dataset, pooled over repeats, split by censoring status and
copula family, each repeat keeping its own validation-derived affine
calibration. Robustness, not headline performance — recovery does not degrade
for censored subjects.

`semi_synthetic_datasets.tex` (dataset characteristics) and
`semi_synthetic_model_summary_tau0{,p5}.tex` (one per dependence level, never
pooled). Table entries are exactly the quantity Figure 1 plots, so a number can
be read off the matching panel. The W/T/L column is a paired oracle-IBS record
against DVFM with a 1e-4 tie tolerance.

---

## 3. Claims to make, and to avoid

**Supported.** Subject-level frailty *ordering* is recovered well and uniformly
(Spearman ~0.75 across datasets, ~0.95 at tau = 0.75 in the synthetic study).
The shared latent pays for itself in prediction above tau ≈ 0.25. DVFM is at or
near the top of the baseline field on all three oracle metrics, and at parity
with HACSurv on joint recovery.

**Not supported.** Dependence-*magnitude* calibration. Every cell
underestimates, four of twelve datasets collapse to independence, and the
estimate barely responds to the true tau. Report it as a limitation with the
event-count threshold, or the heatmap will be read as an unflagged failure.

**Do not claim** a between-copula ordering in either recovery figure — the
within-dataset spread is far smaller than the between-dataset spread.

---

## 4. Known issues

- **DeepSurv results before commit `b74cebe` are invalid.** Its Cox partial
  likelihood broadcast an `(N,)` event mask against an `(N, 1)` risk column, so
  the event indicator cancelled and the model trained as if nothing were
  censored. Fixed, regression test in `tests/test_sota.py`, all DeepSurv numbers
  retuned and rerun.
- **Do not compare Kendall's tau across DVFM and HACSurv.** Both write
  `learned_conditional_kendall_tau` but DVFM takes an empirical tau from 2,000
  sampled (T, C) pairs at the mean covariate while HACSurv evaluates the
  analytic generator integral. Within-model comparisons are fine.
- **`oracle_joint_survival_ise` exists only at tau = 0 and tau = 0.5**, so there
  is no tau sweep for it in the semi-synthetic study.
- **The independence scenario is the only one at tau = 0** — the copula set
  changes with tau, so any code filtering by tau must account for it.
- Figure 1 uses a median with a bootstrap CI while the recovery figures use mean
  +/- 1 SD. Both are defensible; each caption must say which.

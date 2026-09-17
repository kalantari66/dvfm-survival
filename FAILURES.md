# Failure handling

This project treats numerical failure as part of model performance. Failed runs
remain in the result artifacts and are never silently dropped, replaced with a
new seed, or converted into an invented metric value.

## Rank-based comparisons

Semi-synthetic models are evaluated on paired cohorts and splits, so failure is
handled at the atomic seed level:

1. Rank models within each dataset, scenario, seed, and metric. Rank 1 is best.
   Ties receive the minimum tied rank.
2. If a model fails or has a non-finite metric for that seed, assign it one rank
   below the worst successful model in the same paired cell. If several models
   fail, they receive the same failure rank.
3. Average the ten seed-level ranks for each dataset and scenario. Thus, two
   failed seeds contribute two penalties; they do not invalidate the other eight
   successful runs or the entire dataset.
4. For the fixed-scenario rank figure, take each model's median rank across
   datasets. Obtain the 95% non-parametric bootstrap interval by resampling
   datasets, not seeds or subjects.

The required aggregation order is therefore:

`paired seed ranks -> mean over seeds -> median over datasets -> dataset bootstrap`

This policy adapts the failure rule in SurvivalPFN (Qi et al., 2026, Appendices
E.5--E.6), which assigns an invalid model--dataset--metric result one rank below
the worst completed method. Applying the penalty at the seed level is necessary
here because our artifacts retain partial success across ten paired repeats.

## Effect sizes and win/tie/loss summaries

A rank penalty is not a numerical outcome and must not be used as an IBS, CI, or
MAE value. Paired effect sizes use only seeds for which both methods produced a
valid metric, and must report the denominator and failure counts alongside the
estimate (for example, `8/10 valid pairs; 2 DVFM failures`).

A separate conservative reliability summary may count a one-sided failure as a
loss for the failed method and two-sided failure as unresolved. It must be
labelled as failure-aware rather than presented as a metric effect size.

## Reruns and changes after failure

A failed job may be rerun once with the identical frozen configuration and seed
to distinguish a transient infrastructure problem from a deterministic model
failure. A confirmatory failure is not repaired by changing hyperparameters,
selecting a different seed, or choosing a checkpoint using test outcomes. Any
stabilized configuration belongs in a separately identified sensitivity run.

## Current semi-synthetic results

The current aggregate contains 20 DVFM failures: 16 for `latent_dim = 1` and 4
for the `latent_dim = 0` control. Their dataset distribution is:

| Dataset | Failures |
|---|---:|
| FLCHAIN | 11 |
| SUPPORT | 4 |
| WHAS | 2 |
| MIMIC-IV | 2 |
| SEER Brain | 1 |

For the Clayton, target Kendall's tau 0.50 rank figure, the only failures are
DVFM `latent_dim = 1` on FLCHAIN repeats 3 and 9. The other eight FLCHAIN repeats
remain valid and contribute their observed seed-level ranks.

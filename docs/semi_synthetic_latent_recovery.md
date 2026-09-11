# Semi-synthetic frailty recovery outputs

The semi-synthetic SUPPORT runner exports latent recovery by default
(`evaluation.save_latent_recovery: true`), independently of `save_predictions`.
Each selected DVFM checkpoint produces a directory under
`results/semi-synthetic-support/latent_recovery/<scenario>_repeat_<repeat>/`.

For positive Clayton tau, this contains:

- `latent_recovery_validation.csv` and `latent_recovery_test.csv`: original
  cohort row indices, observed event indicators, true standardized log-frailty,
  raw and sign-aligned posterior means, posterior standard deviations,
  calibrated means/stds, and diagnostic 95% intervals and coverage indicators.
- `latent_calibration_metrics.csv`: Pearson, Spearman, R² and RMSE for aligned
  and calibrated estimates, separately for validation/test and all/event/censored
  subjects. Calibrated rows also contain coverage and mean interval width.
- `latent_recovery_metadata.json`: dataset/scenario/seeds/checkpoint, availability,
  validation-derived alignment sign, intercept and slope.

Set `OUTPUT_DIR` in `notebooks/dvfm_frailty_calibration_v2.ipynb` to one of these
run directories. Its expected input filenames and columns are provided directly.
The notebook can then recreate the scatter panels, central-98% view, decile
means/error bars and subgroup statistics. Calibrate each repeat separately;
do not pool validation/test records across fitted models before calibration.

The target is log sampled Gamma frailty, centered and scaled using the generated
cohort (the same convention as the existing synthetic frailty diagnostic).
This is a definition of the simulation target; sign alignment and the fitted
affine mapping use validation subjects only. Test truth never fits that mapping.
Posterior inference uses covariates and observed time/event only. Interval
coverage is a diagnostic, not a guarantee of calibrated Bayesian uncertainty.

Tau zero generates independent uniforms and has no random shared frailty.
Only metadata with `unavailable_no_true_frailty` is exported for that condition.
Scalar latent recovery is supported; other latent dimensions are marked
unavailable. Undefined small/constant-subgroup correlations are stored as NaN.

For additional datasets, generators should populate `SurvivalData.true_z`.
Splitting/preprocessing already preserves it. The dataset-independent
`experiments.latent_recovery.export_latent_recovery` accepts validation/test
`SurvivalData`, original row indices, a fitted model and run context; future
runner branches can use the same `_fit_one_split` export arguments. No SUPPORT
feature or generator assumptions are embedded in this exporter.

Old aggregate logs cannot supply these subject-level outputs; rerun experiments
to generate them. Outputs are written per fitted run, so completed runs retain
their diagnostics if a later condition fails.

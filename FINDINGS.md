# DVFM Findings So Far

## What works

- DVFM reliably recovers subject-level shared frailty when dependence is moderate or strong and the fitted latent dimension is 1 or 5.
- In the Gaussian-frailty pilot, test frailty Spearman correlation was about 0.81 at `kendall_tau = 0.50` and 0.87--0.91 at `kendall_tau = 0.75`.
- The reliable training setup is 200 epochs, learning rate `0.001`, `beta_max = 1`, warmup 50, and a checkpoint chosen by best validation ELBO after epoch 50.
- This post-warmup ELBO checkpoint preserves the recovery achieved by the final model while avoiding numerically unstable late epochs.
- DVFM's predictive advantage is clearest under strong dependent censoring. At `kendall_tau = 0.75`, it beat CoxPH and ClaytonAFT on oracle IBS across every tested censoring level and seed.
- A one-dimensional latent space is the cleanest mechanistic setting. Five dimensions can improve IBS and strong-dependence recovery, but complicate interpretation.

## What does not work consistently

- Good retrospective frailty recovery does not guarantee good prospective prediction: frailty inference uses observed follow-up, whereas prediction for a new subject must integrate over the latent distribution.
- DVFM is not uniformly best. CoxPH is strongest under independence and has the best overall concordance; ClaytonAFT is usually strongest around `kendall_tau = 0.25--0.50`.
- Population dependence recovery remains weak. Learned conditional Kendall's tau is reasonable near 0.25--0.50, but is underestimated near 0.75 and unstable under independence.
- The independence negative control currently fails: DVFM can invent nonzero dependence and loses predictive accuracy relative to CoxPH.
- Reconstruction-NLL checkpointing selects models too early and harms frailty recovery. The older `beta_max = 0.2`, warmup 150, learning rate `0.0005` setup is also unstable.
- Numerical objective failures can coexist with finite-looking predictions, so prediction finiteness alone is not an adequate success check.

## Hyperparameter sweep

- Across seven Gaussian/Clayton scenarios and three paired seeds, `weight_decay = 1e-4` was the only useful default change. Versus the reference, test oracle IBS improved by `0.00044` on average (14/21 cells), while CI, frailty recovery, Kendall's tau error, and runtime were essentially unchanged. The gain is modest and should be confirmed semi-synthetically.
- A wider decoder gave the best dependence recovery, but slightly worsened test IBS and CI; retain it only as a dependence-focused ablation.
- Dropout did not give a balanced improvement. In particular, `dropout = 0.25` harmed IBS, frailty recovery, and dependence recovery despite a small CI gain.
- Batch size 32 was slow and inconsistent; batch size 128 was about twice as fast but predictively worse. Keep batch size 64 for scientific runs and use 128 only for smoke tests if needed.
- Training for 400 epochs doubled runtime for negligible predictive benefit. Keep 200 epochs.
- Aggregate-posterior prediction was marginally worse than prior prediction overall. Use the prior for primary prospective prediction and retain the aggregate posterior only as a diagnostic.

## Latent regularization

- Latent-loading L1 with `alpha = 0.1` was the best overall regularizer: it reduced mean absolute Kendall-tau error from `0.182` to `0.168`, and the independence-cell error from `0.196` to `0.132`, without harming frailty recovery. It is now the default DVFM setting but does not fully solve dependence miscalibration or deactivate the latent dimension.
- The hard-concrete gate failed: its learned value saturated near `1` and worsened dependence calibration. The smooth gate remained open at roughly `0.75--0.81`; it improved the independence cell but worsened strong-dependence recovery. Group lasso was not clearly better than direct latent-loading L1, and combining it with the smooth gate was unstable.

## Current decisions

- Use oracle IBS as the primary synthetic prediction metric, with oracle CI and MAE as secondary metrics. Do not use IBS-Dep for now.
- Report `oracle_joint_survival_ise` for DVFM and HACSurv to evaluate the full conditional event--censoring distribution beyond Kendall's tau.
- Use repeat seeds `0--9`, passing seed `i` independently to DGP generation, sampling, splitting, and model initialization; reuse identical censored cohorts across all compared models.
- Keep train, validation, and test partitions separate; all checkpointing and hyperparameter selection must use validation data only.
- Reject non-finite objectives/gradients and epochs with absolute ELBO, reconstruction NLL, or KL above 100.
- Default DVFM settings for the next synthetic/semi-synthetic experiments: 200 epochs, learning rate `0.001`, batch size 64, `beta_max = 1`, warmup 50, no dropout, encoder `[64, 32]`, decoder `[32, 64]`, `weight_decay = 1e-4`, latent-loading L1 `alpha = 0.1`, and best validation ELBO after warmup.
- Keep latent dimension 1 for interpretable frailty recovery; treat larger latent dimensions as an ablation rather than a default.
- Continue requiring validation IBS improvement, frailty Spearman within 0.03 of the reference, and no material worsening of Kendall's tau error before accepting later changes.

## Synthetic plots

Panel A — individual frailty recovery: recovery improves strongly with the target Kendall’s \(\tau\): roughly \(0.4\!-\!0.6\) at \(\tau=.25\), \(0.75\!-\!0.82\) at \(\tau=.5\), and about \(0.94\) at \(\tau=.75\). More censoring consistently reduces recovery, particularly at weaker dependence. This is expected: censoring removes information about subject-specific frailty.

Panel B — joint-survival benefit: values are \(z=0\) minus \(z=1\) joint-survival ISE, so positive values favor the latent model. At \(\tau=0\), the latent model slightly hurts performance. By \(\tau=.5\) and \(.75\), it improves joint-survival estimation, with gains around \(5\!-\!9\times10^{-3}\). The benefit is generally largest with 75% censoring, where modeling dependence matters most.

Panel C — dependence calibration: the learned dependence is qualitatively correct—it increases as the target increases—but is substantially shrunk toward zero. The model underestimates strong dependence, especially with 75% censoring. At low target dependence, some estimates are negative or spuriously positive, indicating that dependence is difficult to identify when the true signal is weak. The dashed diagonal represents perfect calibration.

Panel D — event-prediction benefit: the same \(z=0\) minus \(z=1\) comparison for Oracle IBS. The latent model provides little benefit at low \(\tau\), but increasingly large benefits at higher \(\tau\). The strongest result is for 75% censoring and \(\tau=.75\): roughly \(0.19\) absolute IBS improvement, i.e. \(19\times10^{-2}\).



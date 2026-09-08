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

## Current decisions

- Use oracle IBS as the primary synthetic prediction metric, with oracle CI and MAE as secondary metrics. Do not use IBS-Dep for now.
- Report `oracle_joint_survival_ise` for DVFM and HACSurv to evaluate the full conditional event--censoring distribution beyond Kendall's tau.
- Use separate DGP, sampling, split, and model seeds, and reuse identical censored cohorts across all compared models.
- Keep train, validation, and test partitions separate; all checkpointing and hyperparameter selection must use validation data only.
- Reject non-finite objectives/gradients and epochs with absolute ELBO, reconstruction NLL, or KL above 100.
- Default DVFM settings for the next synthetic/semi-synthetic experiments: 200 epochs, learning rate `0.001`, batch size 64, `beta_max = 1`, warmup 50, no dropout, encoder `[64, 32]`, decoder `[32, 64]`, `weight_decay = 1e-4`, and best validation ELBO after warmup.
- Keep latent dimension 1 for interpretable frailty recovery; treat larger latent dimensions as an ablation rather than a default.
- Continue requiring validation IBS improvement, frailty Spearman within 0.03 of the reference, and no material worsening of Kendall's tau error before accepting later changes.

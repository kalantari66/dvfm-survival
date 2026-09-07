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

## Current decisions

- Use oracle IBS as the primary synthetic prediction metric, with oracle CI and MAE as secondary metrics. Do not use IBS-Dep for now.
- Use separate DGP, sampling, split, and model seeds, and reuse identical censored cohorts across all compared models.
- Keep train, validation, and test partitions separate; all checkpointing and hyperparameter selection must use validation data only.
- Reject non-finite objectives/gradients and epochs with absolute ELBO, reconstruction NLL, or KL above 100.
- Keep the current recovery-validated configuration as the reference. A new one-factor-at-a-time sweep tests dropout, 400 epochs, batch size, decoder width, and weight decay across Gaussian and Clayton/Gamma frailty data.
- Accept a predictive configuration only if it improves validation oracle IBS while retaining frailty Spearman within 0.03 of the reference and not worsening Kendall's tau error.

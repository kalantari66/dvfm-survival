# Primary Synthetic Experiment

## Purpose

The primary synthetic experiment asks whether DVFM works under its intended
shared-frailty mechanism. It is a mechanistic validation of DVFM, not a
comparison against other survival models.

The experiment tests three questions:

1. Can DVFM recover a known individual frailty from right-censored follow-up?
2. Can DVFM recover the joint event/censoring distribution and its dependence?
3. Can modeling dependent censoring improve recovery of the event-time
   distribution?

The complete specification is `configs/synthetic.yaml`.

## Data-generating process

Each dataset contains 10,000 subjects and 10 independent standard-normal
covariates. A subject has one known scalar shared frailty:

```text
X_i ~ Normal(0, I_10)
z_i ~ Normal(0, 1)
```

Event and censoring times follow Weibull accelerated-failure-time models:

```text
T_i = exp(X_i beta_T + a z_i) [-log(U_Ti)]^(1 / 1.5)
C_i = exp(b_C + X_i beta_C + a z_i) [-log(U_Ci)]^(1 / 1.3)
```

Here, `U_Ti` and `U_Ci` are independent uniforms. The same frailty enters both
time scales and therefore induces dependence between event and censoring time.
The loading `a` is calibrated by simulation to attain the requested conditional
Kendall's tau. The censoring intercept `b_C` is subsequently calibrated to
attain the requested censoring rate.

Only the observed follow-up is supplied to DVFM:

```text
Y_i = min(T_i, C_i)
delta_i = 1(T_i <= C_i)
```

The evaluator retains `T_i`, `C_i`, and `z_i` as oracle quantities. Covariates
are already standard normal, so no additional covariate or time scaling is
applied.

## Experimental grid

- Conditional Kendall's tau: `{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}`.
- Censoring rate: `{0.25, 0.50, 0.75}`.
- Ten repeat seeds, numbered `0` through `9`. Within repeat `i`, seed `i`
  independently initializes the DGP parameters, cohort sampling, data split,
  and model optimization.
- DGP parameters vary across repeats, but remain paired across every dependence
  and censoring condition within a repeat.
- Random 70% training, 10% validation, and 20% test split.

This produces 270 independently sampled datasets. The identical cohort and
split are used for the two DVFM specifications within every condition and seed.

## DVFM specifications

- `latent_dim = 1` is the primary DVFM and matches the true scalar frailty.
- `latent_dim = 0` is an internal DVFM conditional-independence control. It is
  not treated as a competing method.

Both use encoder widths 64--32, decoder widths 32--64, batch size 64, learning
rate `0.001`, weight decay `1e-4`, no dropout, and 200 epochs. Decoder weights
carrying the shared latent into the event and censoring margins receive the
default L1 penalty `latent_loading_l1 = 0.1`; this coefficient is an explicit
hyperparameter and the `latent_dim = 0` control incurs no latent-loading
penalty. The KL weight increases to `beta_max = 1` over the first 50 epochs.

The primary checkpoint is the numerically valid epoch with the best validation
ELBO at or after epoch 50. Test data are not used for checkpoint selection.
Together, the 270 datasets and two DVFM specifications produce 540 fits.

## Evaluation

### Individual frailty recovery

For `latent_dim = 1` and positive target dependence, the posterior latent is
compared with the known `z_i`. Its sign and affine scale are determined using
validation subjects only and then applied unchanged to test subjects.

Metrics are reported for all test subjects, observed-event subjects, and
censored subjects:

- Frailty Spearman correlation, the primary rank-recovery metric.
- Frailty Pearson correlation.
- Validation-calibrated RMSE.
- Validation-calibrated R-squared.

At target tau zero, `z_i` does not affect either time and frailty recovery is
therefore undefined rather than expected to equal zero.

### Dependence and joint-distribution recovery

- Learned conditional Kendall's tau and its signed and absolute target error.
- `oracle_joint_survival_ise`, comparing learned and true
  `P(T > t, C > c | X)` over 128 held-out subjects and a 15 by 15 time grid.

Kendall's tau tests overall dependence strength. Joint-survival ISE tests the
full joint surface, including both marginal distributions.

### Event-time prediction

Oracle CI, IBS, MAE, and calibration compare predicted event survival against
the retained uncensored `T_i`. Results are produced from both the prior and the
aggregate-posterior predictive distributions. The paired `latent_dim=1` versus
`latent_dim=0` result indicates whether DVFM's shared latent path improves
event-time recovery under dependent censoring.

## Outputs and execution

Results are written to `results/synthetic/`. Compact CSV artifacts contain raw
and aggregated metrics, training histories, calibration curves, frailty
diagnostics, numerical failures, seeds, and the resolved configuration. Full
per-subject prediction arrays are not saved.

Run the complete experiment as one GWF target:

```bash
gwf -f workflows/synthetic/workflow.py status
gwf -f workflows/synthetic/workflow.py run
```

The target requests one H200 GPU, four CPU cores, 25 GB memory, and 12 hours.
Expected runtime is approximately seven to nine hours.

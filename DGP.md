# Data-generating processes

This document defines the data-generating processes (DGPs) used by the
controlled synthetic and semi-synthetic experiments. It distinguishes the
generating estimands from model-fitting choices and records the latent variable
used by subject-level recovery analyses.

## Common survival notation

For subject (i), let (T_i) be the event time, (C_i) the censoring time,
(Y_i = \min(T_i, C_i)) the observed duration, and
(\Delta_i = 1(T_i \le C_i)) the event indicator. The experiments retain both
(T_i) and (C_i), so prediction can be evaluated against oracle event-time
truth even under dependent censoring.

Randomness is separated into DGP-parameter, cohort-sampling, split, and model
seeds where the configuration provides separate streams. All models within a
scenario receive the same generated subjects and split.

## Controlled synthetic experiment

The primary controlled experiment is configured by `configs/synthetic.yaml`
and uses the shared-Gaussian-frailty generator in `src/utility/synthetic.py`.
It is designed to identify whether DVFM recovers a known scalar dependence
mechanism, rather than to mimic a particular clinical cohort.

### Covariates and marginal parameters

Covariates are sampled independently as

\[
X_i \sim N(0, I_p).
\]

The event and censoring coefficient vectors are sampled once from
(N(0, 0.25^2)) using the DGP seed. Weibull shapes are fixed at
(k_T=1.5) and (k_C=1.3). A cohort uses one shared standard-normal latent
(Z_i\sim N(0,1)) and independent uniforms (U_{Ti},U_{Ci}\).

### Shared-Gaussian-frailty times

Event and unshifted censoring times are

\[
T_i = \exp(X_i^T\beta_T + \lambda Z_i)
      [-\log U_{Ti}]^{1/k_T},
\]

\[
C_i^{(0)} = \exp(X_i^T\beta_C + \lambda Z_i)
      [-\log U_{Ci}]^{1/k_C}.
\]

The common loading (\lambda\) is calibrated by deterministic Monte Carlo and
bisection so that the event/censor residuals attain the requested conditional
Kendall's tau. The recovery target is the generating (Z_i); its sign is not
identified by the fitted latent model, so alignment uses validation subjects
only.

The censor intercept (b_C) is selected from the generated sample:

\[
b_C = Q_{1-r}\{\log T_i-\log C_i^{(0)}\}, \qquad
C_i=\exp(b_C)C_i^{(0)},
\]

where (r) is the target censoring fraction. This gives the requested
finite-sample censoring rate up to ties/rounding.

### Clayton/Gamma mechanism diagnostic

Mechanism-comparison and frailty-recovery configurations also use a
Marshall--Olkin Clayton DGP. For target tau,

\[
\theta = \frac{2\tau}{1-\tau}, \qquad
W_i\sim \operatorname{Gamma}(1/\theta,1),
\]

and, with independent unit exponentials (E_{Ti},E_{Ci}),

\[
U_{ji}=(1+E_{ji}/W_i)^{-1/\theta},\quad j\in\{T,C\}.
\]

These uniforms enter Weibull regression margins. The known recovery target is
the cohort-standardized (\log W_i). This diagnostic reproduces the shared
Gamma mechanism, but its synthetic Weibull margins are not the semi-synthetic
Cox margins described below.

### Synthetic evaluation

The primary synthetic grid varies tau and the censoring fraction while holding
the structural mechanism known. It evaluates event-time prediction, dependence
recovery, joint-survival recovery, and subject-level latent recovery. The
matched `latent_dim=0` fit is an internal mechanistic control; the
semi-synthetic study instead evaluates the final scalar-latent model against
external comparators.

## Semi-synthetic experiment

The semi-synthetic experiment is configured by
`configs/semi_synthetic.yaml` and implemented in
`src/utility/semisynthetic.py`. It preserves covariates and marginal survival
patterns from real cohorts while replacing event and censoring times with
fully known draws.

### Source-cohort preparation

For each dataset, rows with non-positive or non-finite durations are removed.
Configured large cohorts may be deterministically subsampled while stratifying
on event status and observed-time quantiles. Numeric covariates are imputed and
standardized; categorical covariates are imputed and one-hot encoded. These
cohort-level features are used to define the DGP.

Two ridge-penalized Cox models are fitted to the complete prepared source
cohort:

- the event margin uses the source event indicator;
- the censor margin uses one minus the source event indicator.

Both use the configured ridge penalizer, currently 0.01. If
(H_{0j}(t)) is a fitted Breslow baseline cumulative hazard and
(r_j(X)=\exp(X^T\hat\beta_j)), the conditional survival margin is

\[
S_j(t\mid X)=\exp[-H_{0j}(t)r_j(X)].
\]

Given a survival-uniform draw (U_j), inverse transformation solves

\[
H_{0j}(t)=-\log(U_j)/r_j(X).
\]

The empirical cumulative hazard is interpolated, with linear tail
extrapolation beyond its final positive increment to avoid placing all extreme
draws at the last observed time.

### Copula mechanisms

The sampled pair ((U_T,U_C)) consists of conditional-survival uniforms. At
tau zero the experiment uses one product-copula condition, with no shared
latent recovery target.

#### Gaussian

Kendall's tau is converted to the latent-normal correlation

\[
\rho=\sin(\pi\tau/2).
\]

For positive tau, sample a common factor (Z\sim N(0,1)) and independent
(\epsilon_T,\epsilon_C\sim N(0,1)):

\[
G_T=\sqrt\rho Z+\sqrt{1-\rho}\epsilon_T,\qquad
G_C=\sqrt\rho Z+\sqrt{1-\rho}\epsilon_C,
\]

\[
U_T=\Phi(G_T),\qquad U_C=\Phi(G_C).
\]

This is distributionally identical to sampling a bivariate normal with
correlation matrix `[[1, rho], [rho, 1]]`, but it retains the symmetric common
factor. The recovery target is cohort-standardized (Z).

#### Clayton

For (\theta=2\tau/(1-\tau)), sample

\[
W\sim\operatorname{Gamma}(1/\theta,1),\qquad
U_j=(1+E_j/W)^{-1/\theta}.
\]

The recovery target is cohort-standardized (\log W).

#### Frank

For positive tau, (\theta>0) is obtained numerically from

\[
\tau=1-\frac{4}{\theta}+\frac{4D_1(\theta)}{\theta},
\]

where (D_1) is the first Debye function. Let
(p=1-\exp(-\theta)), sample a logarithmic-series mixing variable
(N\sim\operatorname{LogSeries}(p)), and independent unit exponentials
(E_T,E_C). With Frank generator

\[
\psi(t)=-\frac{1}{\theta}
\log\{1-(1-\exp(-\theta))\exp(-t)\},
\]

the uniforms are

\[
U_j=\psi(E_j/N).
\]

This Marshall--Olkin representation is equivalent to direct conditional Frank
sampling for positive dependence, but exposes the generating shared variable.
The recovery target is cohort-standardized (\log N). Because (N) is
discrete, rank-recovery statistics can contain ties.

#### Gumbel

Set (\theta=1/(1-\tau)) and (\alpha=1/\theta). The
Marshall--Olkin representation samples a positive alpha-stable variable (W)
and independent unit exponentials, then uses

\[
U_j=\exp[-(E_j/W)^\alpha].
\]

The implementation samples (W) with the standard positive-stable angular
construction. The recovery target is cohort-standardized (\log W).

### Time generation and censoring calibration

For fixed real-cohort covariates (X_i), the copula uniforms are transformed
through the fitted margins:

\[
T_i=S_T^{-1}(U_{Ti}\mid X_i),\qquad
C_i^{(0)}=S_C^{-1}(U_{Ci}\mid X_i).
\]

To obtain target censoring fraction (r), define

\[
s=Q_{1-r}\{T_i/C_i^{(0)}\},\qquad C_i=sC_i^{(0)}.
\]

Thus censoring occurs when (T_i/C_i^{(0)}>s), and the requested rate is met
to within at most one subject for continuous draws. Every fitted model sees the
same (X_i,Y_i,\Delta_i,T_i,C_i) within a dataset/scenario/repeat.

### Recovery and evaluation targets

Subject-level `true_z` is available for every positive-dependence family:

| Copula | Stored recovery target |
|---|---|
| Gaussian | standardized common normal (Z) |
| Clayton | standardized (\log W_{\text{Gamma}}) |
| Frank | standardized (\log N_{\text{LogSeries}}) |
| Gumbel | standardized (\log W_{\text{stable}}) |
| Independence | unavailable by definition |

For compatibility, the internal `clayton_theta` field and result-table
`Theta` column retain their historical names. Their value is family-specific:
rho for Gaussian, theta for Clayton/Frank/Gumbel, and zero for independence.

The term **shared latent** is preferred in cross-family plots; only the
positive Archimedean variables are literally positive mixing frailties.
Posterior sign alignment and affine calibration are fitted on validation
subjects and then frozen before test evaluation.

The generated event time supplies oracle IBS, concordance, MAE, and calibration.
The known copula/margins supply oracle joint-survival ISE for models that expose
a joint event/censor surface. Observed-data IPCW metrics are secondary because
their independent-censoring assumption need not hold in dependent conditions.

Before fitting any model, observed train/validation/test durations and query
grids are divided by the training median duration. Predicted medians are mapped
back to the source time unit, and all reported metrics and joint-surface
integrals use the original time scale. This model-side normalization does not
alter the DGP.

## Reproducibility consequences

Changing a copula sampler while preserving its population copula can still
change the realized cohort for a fixed seed because random numbers are consumed
differently. The explicit-factor Gaussian and Frank implementations therefore
require regenerated semi-synthetic results. Old aggregate tables cannot recover
the newly retained generating variables retrospectively.

No external copula package is required: the implemented constructions are
short, seeded through NumPy's `Generator`, and expose the exact latent variable
needed by the recovery estimand.

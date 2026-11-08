# Rebuttal experiments: results per task

Every run below uses the paper's configs, seeds, splits and per-dataset tuned hyperparameters, and no model was retuned. Each section answers one item from the task list: what we ran, what the numbers show, and how far we can lean on the result. Code is on branch `rebuttal`; outputs are in `rebuttal/<task>/`.

## What DVFM provides that the baselines do not

The rebuttal experiments make two benefits of DVFM concrete. Neither is available from a global copula (HACSurv), an event-only frailty model (Cox-Gamma) or a model with independent latents. Both deserve a place in the revised paper; the covariate-varying experiment would be new.

**1. Dependence that varies with covariates, predicted for a new patient from x alone.** DVFM's frailty can act differently depending on covariates, so the strength of event–censoring dependence can differ between patients. In the test DGP, an unobserved frailty z enters both log event and log censoring time with an observed-covariate loading λ(x1), so the dependence (Kendall's τ between T and C given x) rises with x1. The exact definitions are in the λ(X) section. DVFM's τ(x) follows this rise; HACSurv's single copula cannot.

| Smooth DGP, x1 | −2 | −1 | 0 | 1 | 2 |
|---|---|---|---|---|---|
| True τ(x) | 0.00 | 0.02 | 0.26 | 0.46 | 0.49 |
| DVFM τ(x), median over 10 repeats | 0.15 | 0.18 | 0.29 | 0.39 | 0.42 |
| HACSurv τ (one global value) | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 |

τ(x) rises with x1 in 9 of 10 repeats (median correlation with the truth 0.93). With a step in dependence at x1 = 0, DVFM puts the higher τ on the correct side in 10 of 10 repeats. Details are in the λ(X) section below.

**2. Subject-level answers, for example "is this censored patient high-risk?".** Because the latent is shared, a patient's censoring time carries information about their frailty. DVFM uses it; models without a shared latent cannot. The test below takes censored test patients only, at τ = 0.5, and asks how many of those a model places in its top quarter of risk really are in the top quarter of the true frailty. Chance is 25%.

| Censored patients: precision for the top-quarter frailty | DVFM (paper, d_z=1) | Shared latents (two-trunk, d_z=2) | Independent latents (two-trunk, d_z=2) |
|---|---|---|---|
| Median over all 12 datasets | 67% (range 50–77%) | — | — |
| GBSG / METABRIC / SUPPORT / SEER liver | 66% / 69% / 67% / 75% | 66% / 71% / 67% / 75% | 55% / 64% / 58% / 65% |

On the same censored subjects, DVFM's posterior mean also tracks the true frailty better than the Cox-Gamma posterior frailty on 11 of 12 datasets (median Spearman gain +0.19), and better than independent latents on all four control datasets (+0.09 to +0.18).

## P0 Prior versus aggregate-posterior prediction (SGkb, kyMv, FpZd)

The reviewers asked whether DVFM's predictions depend on drawing the latent from the aggregate posterior q_agg instead of the N(0,1) prior. To answer this cleanly, we refit all 600 semi-synthetic DVFM models (12 datasets × 5 scenarios × 10 seeds) and 120 synthetic ones. We scored every fitted model twice, with the same checkpoint, the same Monte Carlo draws and the same test set; only the mixing distribution differed. As a side result, the refits reproduce the paper's numbers bit-for-bit in all 600 cells, so the semi-synthetic benchmark is fully reproducible from the released code.

| Metric (24 dataset × τ cells) | Median Δ (q_agg − prior) | q_agg significantly better | q_agg significantly worse |
|---|---|---|---|
| Oracle IBS | +0.00004 (max +0.0011) | 2 | 7 |
| Oracle C-index, MAE | ≤ 0.3% relative | 2–3 | 0–1 |
| Joint-survival ISE | +2.1% (max +5.8%) | 3 | 16 |

**What the table shows.**
- **Marginal accuracy is unaffected.** The largest IBS change is 0.001, below a third of the seed-to-seed spread, and C-index and MAE move by at most 0.3%.
- **The joint is slightly worse under q_agg**, in two thirds of the cells, by a median of 2%.
- **The synthetic grid agrees.** Its only exception is a small C-index gain for q_agg at τ = 0.75.
- **Why the two coincide:** under the model z is independent of x, so q_agg is the plug-in estimate of the prior. The trained q_agg is indeed close to N(0,1), with mean ≈ 0 and SD 0.95–1.0 on every dataset.

The paper uses different mixing distributions for different quantities, and the text should make this explicit:

| Quantity | Mixing distribution |
|---|---|
| Survival curves, event-time inference | q_agg |
| Joint log p(t, δ \| x) (ELBO, IWAE) | Prior; the encoder is only the importance-sampling proposal |
| Event- and censoring-margin likelihoods | q_agg |
| Joint ISE, learned τ | Prior |

On cost, q_agg is one encoder pass over the training split and can be built once and cached. The released code rebuilds it on every prediction call, which is wasteful but still cheap:

| q_agg cost at N = 10,000 | CPU | GPU |
|---|---|---|
| Build once | 0.05 s | 0.11 s |
| Predict 1,000 patients (cached; independent of N) | 0.06 s | 0.06 s |
| One patient: cached / rebuilt per call (released code) | 12 / 49 ms | 52 / 129 ms |

**If the revision switches to the prior.** None of the semi-synthetic conclusions change:
- the d_z = 0 ablation still shows lower IBS on 7 of 12 datasets, with the same four improving by 10–25%;
- SEER brain and SEER liver are worse by 7–8% instead of 8%;
- DVFM's mean rank stays 3.1.

The same holds on real data and on the covariate-varying DGP, where we refit and scored every model under both mixing distributions:

| Prior vs q_agg, identical fits | Largest difference | Conclusion changes? |
|---|---|---|
| PRO-ACT (10 seeds × d_z ∈ {0, 1}): D-calibration, KM-calibration, C-index, IBS-IPCW, event- and censoring-margin likelihoods | Means: D-cal statistic 7.97 vs 8.06, margin log-likelihoods within 0.0002. Per seed: C-index ≤ 0.006, margin log-likelihoods ≤ 0.003. Seeds passing D-calibration: 9 of 10 under both. | No |
| PRO-ACT q_agg itself | Mean ≈ 0, SD 0.95–0.98 | It is the prior, to within 5% |
| Covariate-varying DGP, oracle IBS | 0.1140 vs 0.1136 (step); 0.1239 vs 0.1247 (smooth) | No |

On real data, too, q_agg is essentially N(0,1), so switching the revision to prior-based prediction changes no result in the paper or the rebuttal. Every refit used for this check reproduces the original runs exactly.

## P0 Covariate-varying dependence λ(X) (kyMv, FpZd)

The paper says DVFM can represent dependence that varies with covariates, and reviewers asked us to show this at prediction time.

**Data-generating process.** We use the paper's Gaussian shared-frailty DGP (N = 10,000, 10 covariates, 50% censoring, 10 repeats), changed in one respect: the frailty's effect depends on x1. For each subject:

- Observed covariates x ~ N(0, I₁₀); unobserved frailty z ~ N(0, 1); independent noise U_E, U_C ~ Uniform(0, 1).
- Event time: T = exp(xᵀβ_E + λ(x)·z) · (−log U_E)^(1/1.5).
- Censoring time: C = exp(b + xᵀβ_C + λ(x)·z) · (−log U_C)^(1/1.3).

These are Weibull times with shapes 1.5 and 1.3 and scales exp(xᵀβ + λ(x)·z). The intercept b is calibrated to 50% censoring, and we observe min(T, C) and the indicator 1[T ≤ C].

**Where the dependence comes from.** Given both x and z, T and C are independent. Given only x, they are dependent, because the shared z moves both, and its loading λ(x) sets how strongly:

| Version | Frailty loading λ(x) | Dependence |
|---|---|---|
| Step | λ_max · 1[x1 ≥ 0] | None for x1 < 0; τ = 0.5 for x1 ≥ 0 |
| Smooth | λ_max · σ(2·x1), with σ(u) = 1/(1 + e^(−u)) | Rises continuously from τ ≈ 0 at low x1 to τ ≈ 0.5 at high x1 |

λ_max is calibrated so that a subject with the full loading has conditional Kendall's τ = 0.5. So z causes the dependence and x1 controls its strength; x2 to x10 only shift the times. In the example of the opening section, x1 could be disease severity and z an unmeasured frailty that makes severe patients both die and drop out sooner.

**What is measured at a reference patient x_ref** (training mean of x, with x1 set to −2, −1, 0, 1 or 2):

| Quantity | Definition |
|---|---|
| True τ(x_ref) | Kendall's τ between T and C given x = x_ref. It depends on x only through λ(x) (Monte Carlo, 200,000 draws). |
| DVFM τ̂(x_ref) | Kendall's τ of 2,000 pairs (T⁽ᵐ⁾, C⁽ᵐ⁾) drawn from the fitted decoder p_θ(t, c \| x_ref, z⁽ᵐ⁾) with z⁽ᵐ⁾ ~ N(0, 1). It uses only x_ref, never an outcome. |
| HACSurv τ̂ | The Kendall's τ of its single fitted copula, the same for every x |
| DVFM latent recovery | Spearman correlation between the posterior mean E_q[z \| xᵢ, tᵢ, δᵢ] (sign aligned on validation) and the true zᵢ, for test subjects with x1 near x_ref |

DVFM can represent a τ that varies with x because its decoder takes (x, z) jointly, so the network can learn how much z matters for each x (an x × z interaction). DVFM uses the synthetic study's settings unchanged. HACSurv, the strongest copula baseline in the paper, was run with its pilot settings (untuned).

| Smooth DGP, x1 | −2 | −1 | 0 | 1 | 2 |
|---|---|---|---|---|---|
| True τ | 0.00 | 0.02 | 0.26 | 0.46 | 0.49 |
| DVFM τ(x), median over 10 repeats¹ | 0.15 | 0.18 | 0.29 | 0.39 | 0.42 |
| HACSurv τ (global copula) | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 |
| DVFM latent recovery (Spearman), median | 0.05 | 0.20 | 0.55 | 0.72 | 0.79 |

| Step DGP | True τ | DVFM τ, median¹ | HACSurv τ | DVFM latent recovery, median |
|---|---|---|---|---|
| x1 < 0 (no dependence) | 0 | 0.21 | 0.18 | 0.00 |
| x1 ≥ 0 (τ = 0.5) | 0.5 | 0.41 | 0.18 | 0.78 |

¹ In 3 of 10 repeats the fit settles on a negative-dependence solution with an equally good validation ELBO, an instance of the weak identifiability discussed in Section 4. The same repeats do this with α = 0.

**What this shows.** DVFM picks up where in covariate space event and censoring are dependent, and it does so for a new patient described only by x.
- *Smooth DGP:* τ(x) rises with x1 in 9 of 10 repeats, with a median correlation of 0.93 with the true curve.
- *Step DGP:* the dependent stratum gets the higher τ in 10 of 10 repeats.
- *Latent recovery:* it follows the dependence, near zero where the latent has no effect and close to 0.8 where it drives both times.
- *Against HACSurv:* a global-copula model cannot express any of this; its single τ of about 0.2 is too high in the independent region and too low in the dependent one.

| Oracle IBS | DVFM (prior) | DVFM (q_agg) | HACSurv (untuned) |
|---|---|---|---|
| Step | 0.1140 | 0.1136 | 0.1024 |
| Smooth | 0.1239 | 0.1247 | 0.1032 |

HACSurv's lower IBS here comes from its margins rather than from dependence modelling: its lead is just as large in the region with no dependence (step: 0.066 vs 0.080 for x1 < 0, DVFM under the prior). Cox-Gamma and CoxPH were not run on this DGP; neither can represent dependence that varies with x.

## P0 Matched independent-latent control (SGkb)

SGkb asked whether DVFM's gains come from sharing a latent between event and censoring, or simply from the extra flexibility a latent mixture gives each margin. We built two decoders that differ only in that respect. Both have two trunks and d_z = 2:
- **Shared:** both latents feed both trunks.
- **Independent:** z₁ feeds only the event trunk and z₂ only the censoring trunk, so each margin is still a Weibull mixture but event and censoring are conditionally independent.

Capacity, latent budget, training, evaluation and seeds are identical, and both use the tuned d_z = 1 hyperparameters without retuning. We ran 4 datasets × 5 scenarios × 10 seeds (400 fits).

| Joint ISE, τ = 0.5 | Shared d_z=2 | Independent d_z=2 | DVFM d_z=1 | d_z=0 | Shared vs independent |
|---|---|---|---|---|---|
| GBSG | **0.0120** | 0.0156 | 0.0131 | 0.0183 | Shared better (CI excludes 0) |
| METABRIC | **0.0068** | 0.0072 | 0.0079 | 0.0091 | Shared marginally better |
| SUPPORT | 0.0120 | 0.0121 | **0.0115** | 0.0135 | Tied |
| SEER liver | 0.0126 | **0.0117** | 0.0135 | **0.0117** | Independent better |

**Accuracy.** Even the independent latent improves joint ISE over the latent-free model on three of four datasets and ties on SEER liver. So part of DVFM's gain over d_z = 0 comes from the richer Weibull-mixture margins a latent provides, and the independent control obtains most of it on METABRIC and SUPPORT. Sharing adds a further gain in joint ISE on GBSG (clearly) and on METABRIC (marginally). It ties on SUPPORT and is behind on SEER liver. On IBS the two variants are within 0.002 everywhere except SEER liver.

**What sharing contributes beyond accuracy.** Sharing is what lets a subject's censoring time carry information about their frailty, and the subject-level recovery shows this directly:

| Spearman with true z, τ = 0.5 (test set) | Censored: shared d_z=2 | Censored: DVFM d_z=1 | Censored: independent d_z=2 | Observed events: shared | Observed events: independent |
|---|---|---|---|---|---|
| GBSG | **0.66** | 0.65 | 0.48 | 0.53 | 0.38 |
| METABRIC | **0.77** | 0.76 | 0.68 | 0.77 | 0.76 |
| SUPPORT | **0.71** | **0.71** | 0.56 | 0.71 | 0.71 |
| SEER liver | **0.83** | **0.83** | 0.69 | 0.75 | 0.75 |

On censored subjects, the shared latent recovers the true frailty substantially better on all four datasets, by 0.09–0.18. On subjects whose event was observed the two are equal, except on GBSG. This is the paper's mechanism: censoring is informative about the latent only when the latent is shared. The independent model also cannot produce a dependence estimate (its learned τ is about 0 by construction), joint predictions that reflect dependence, or covariate-varying τ(x).

**Caveat.** The two-trunk models ran on CPU and the d_z = 1 refits on GPU. Shared vs independent is exactly like-for-like; comparisons with d_z = 1 hold only within seed noise.

## P0 Training and prediction wall-clock times (FpZd, SGkb)

FpZd asked for training and prediction times under comparable hardware. We timed all three models on the same machine in each setting, with 4 threads and a warm-up before timing:
- **Data:** Clayton τ = 0.5 cells on three datasets of increasing size; 3 seeds; median shown.
- **Hardware:** an AMD EPYC 9655 CPU node and an NVIDIA L40S GPU node.
- **Precision:** each method's default (DVFM fp32; HACSurv and CG fp64).

| Training (s) | DVFM CPU | HACSurv CPU | CG CPU | DVFM GPU | HACSurv GPU | CG GPU |
|---|---|---|---|---|---|---|
| GBSG (n_train 479) | 4.2 | 24.2 | 0.10 | 9.1 | 19.3 | 0.30 |
| SUPPORT (6,210) | 51.8 | 150.7 | 0.73 | 108.3 | 112.7 | 1.50 |
| SEER liver (7,000) | 68.5 | 341.3 | 0.82 | 134.2 | 234.7 | 1.54 |

| Prediction, SEER liver | DVFM CPU | DVFM GPU | HACSurv CPU | HACSurv GPU | CG CPU |
|---|---|---|---|---|---|
| 1,000 patients (q_agg cached; amortized per patient) | 0.057 s (0.06 ms) | 0.055 s (0.06 ms) | 0.116 s | 0.034 s | 0.0003 s |

**What we see.**
- **DVFM vs HACSurv:** DVFM trains 3–6× faster on CPU and 1–2× faster on GPU. Per epoch the two cost the same (0.34 s on SEER liver, CPU); the difference is that DVFM trains for a fixed 200 epochs while HACSurv typically runs 550–1,000 before early stopping.
- **DVFM on GPU:** it is about 2× slower than on CPU, because its networks are small and batch size 64 leaves the GPU waiting on kernel launches. For this model the CPU is the right hardware.
- **CG:** a linear frailty model, 40–90× faster than DVFM to train. We should concede that.

Prediction cost is negligible for all three.

## P1 Marginal-flexibility control against HACSurv (FpZd)

FpZd asked whether DVFM's advantages over HACSurv come from dependence modelling or from its more flexible margins.

**DVFM vs HACSurv overall.** Across all 12 datasets in the paper's runs, DVFM has the lower median joint ISE and IBS in every copula family. This includes the Archimedean families that HACSurv's copula is built for.

| τ = 0.5, all 12 datasets | Joint ISE: DVFM better on | Median joint ISE, DVFM vs HACSurv | IBS: DVFM better on | Median IBS, DVFM vs HACSurv |
|---|---|---|---|---|
| Clayton | 8 of 12 | 0.0123 vs 0.0130 | 6 of 12 | 0.110 vs 0.115 |
| Frank | 7 of 12 | 0.0137 vs 0.0171 | 5 of 12 | 0.111 vs 0.116 |
| Gaussian | 8 of 12 | 0.0135 vs 0.0139 | 6 of 12 | 0.111 vs 0.116 |
| Gumbel | 8 of 12 | 0.0138 vs 0.0157 | 5 of 12 | 0.110 vs 0.117 |

**Controlled comparison.** To separate margins from dependence, we compared HACSurv with the independent-latent model from the matched control above. That model has DVFM's Weibull-mixture margins but no shared dependence. The comparison covers the four control datasets with the same seeds and splits:

| τ = 0.5 | Joint ISE: independent | Joint ISE: DVFM | Joint ISE: HACSurv | IBS: independent | IBS: DVFM | IBS: HACSurv |
|---|---|---|---|---|---|---|
| GBSG | 0.0156 | 0.0131 | 0.0385 | 0.1487 | 0.1482 | **0.1372** |
| METABRIC | 0.0072 | 0.0079 | **0.0063** | 0.1612 | 0.1601 | 0.1610 |
| SUPPORT | 0.0121 | **0.0115** | 0.0142 | 0.1190 | 0.1184 | **0.1165** |
| SEER liver | 0.0117 | 0.0135 | **0.0075** | **0.0819** | 0.0886 | 0.0823 |

**What we see.** Both components contribute.
- *The margins:* the flexible Weibull-mixture margins are a real strength. On GBSG and SUPPORT the no-dependence variant already beats HACSurv on joint ISE, so DVFM's margins alone outperform HACSurv's monotone neural margins there.
- *The shared latent:* it improves joint ISE further on GBSG, and it drives the much better recovery of subject-level frailty on censored subjects (previous section).
- *HACSurv:* it leads on METABRIC and SEER liver in this subset. These are two of the four datasets (with EMPLOYEE and SEER stomach) where it beats DVFM on joint ISE overall, so the four-dataset subset is less favourable to DVFM than the full benchmark.

## P1 Identification theorem scope and matched decoder variant (kyMv)

kyMv pointed out that the identification theorem covers a restricted multiplicative frailty model, while the implemented DVFM lets the latent affect both the Weibull shape and scale. This is correct. `semi_synthetic.yaml` sets neither `latent_path` nor `shape_mode`, so the paper's DVFM runs the defaults: a nonlinear latent path and a z-dependent shape ([runner.py:170-171](../src/experiments/runner.py#L170-L171)). That model is outside the theorem's class, and the text must say so.

To see whether staying inside the class would help, we ran a decoder that satisfies its structure: log-scale f(x) + g·z with one global Weibull shape per margin. It still learns the baseline hazards instead of assuming them known. We ran it with the paper's α = 0.1 and with α = 0, because in this decoder the ℓ1 penalty acts directly on the frailty loading g. Everything else matches the paper fits, seed for seed. The synthetic DGP (Weibull, global shape, log-scale additive in x and z) lies exactly inside this class, so the restricted decoder is correctly specified there and the paper decoder is over-flexible.

| Synthetic grid (mean over 3 censoring levels × 10 repeats) | True τ = 0 | 0.25 | 0.5 | 0.75 |
|---|---|---|---|---|
| Learned τ: paper decoder | −0.14 | 0.22 | 0.46 | 0.52 |
| Learned τ: theorem class, α = 0.1 | −0.23 | 0.21 | 0.43 | 0.19 |
| Learned τ: theorem class, α = 0 | −0.29 | 0.15 | 0.50 | 0.26 |
| \|τ error\|: paper / α = 0.1 / α = 0 | 0.36 / 0.29 / 0.46 | 0.28 / 0.23 / 0.39 | 0.09 / 0.10 / 0.06 | 0.23 / 0.56 / 0.49 |
| Oracle IBS: paper / α = 0.1 / α = 0 | 0.128 / 0.104 / 0.116 | 0.128 / 0.121 / 0.134 | 0.110 / 0.110 / 0.107 | 0.084 / 0.119 / 0.111 |
| Joint ISE: paper / α = 0.1 / α = 0 | 0.0064 / 0.0026 / 0.0035 | 0.0035 / 0.0025 / 0.0031 | 0.0008 / 0.0013 / 0.0009 | 0.0003 / 0.0034 / 0.0026 |
| Latent recovery (Spearman): paper / α = 0.1 / α = 0 | — | 0.51 / 0.49 / 0.47 | 0.81 / 0.81 / 0.82 | 0.94 / 0.91 / 0.93 |

**Synthetic grid: correct specification does not buy better dependence estimates.**
- *Overall τ error:* averaged over the grid it is 0.24 for the paper decoder, 0.29 for the restricted decoder at α = 0.1, and 0.35 at α = 0.
- *Strong dependence (τ = 0.75):* the restricted decoder underestimates τ badly, at 0.19–0.26, and its IBS and joint ISE degrade with it.
- *No dependence (τ = 0):* at α = 0 it invents a consistent negative τ of about −0.5 under 50% and 75% censoring, with very little spread between repeats.
- *Weak or no dependence:* its one advantage is at τ ≤ 0.25. There its joint ISE is better at both α, and its IBS is better at α = 0.1 (at α = 0, only at τ = 0). That is what one expects when the marginal shape is fixed correctly, and it is a marginal effect rather than a dependence effect.
- *τ = 0.5:* the three decoders are equivalent.
- *Collapse:* at α = 0.1, 8 of 120 fits switch the latent off completely, all at τ ≤ 0.25; at α = 0 none do.
- *Diagnosis:* the decoder is correctly specified here, so the τ = 0.75 shortfall points to an optimisation problem. The single loading g has to grow large from a small initial value against the KL term, and it does not get there.

| Semi-synthetic, τ = 0.5 | Joint ISE: paper d_z=1 / α = 0.1 / α = 0 / d_z=0 | IBS: paper / α = 0.1 / α = 0 / d_z=0 | Latent collapsed (of 50 fits): α = 0.1 / α = 0 |
|---|---|---|---|
| GBSG | **0.0131** / 0.0210 / 0.0202 / 0.0183 | 0.148 / 0.154 / 0.154 / **0.146** | 42 / 17 |
| METABRIC | **0.0079** / 0.0091 / 0.0107 / 0.0091 | **0.161** / 0.162 / 0.164 / 0.165 | 0 / 0 |
| SUPPORT | **0.0115** / 0.0161 / 0.0151 / 0.0135 | 0.118 / 0.120 / 0.119 / **0.118** | 40 / 0 |
| SEER liver | 0.0135 / 0.0200 / 0.0197 / **0.0117** | 0.088 / 0.095 / 0.095 / **0.082** | 0 / 0 |

**Semi-synthetic: the restricted decoder is worse everywhere.**
- *Joint ISE:* worse than the paper decoder on all four datasets, and often worse than having no latent at all.
- *IBS:* worse on all four datasets, by 0.6–9%.
- *Latent use:* at α = 0.1 it switches the latent off in most GBSG and SUPPORT fits. Removing the penalty prevents this on SUPPORT, but not on GBSG.
- *Latent recovery:* also lower, e.g. on censored subjects 0.28–0.52 against 0.65 for the paper decoder on GBSG.

**Conclusion.** The paper should state plainly that the theorem covers the restricted known-hazard multiplicative model and gives no guarantee for the neural decoder. It should not suggest that fitting the restricted model would be safer: in our runs the restricted decoder recovers dependence no better even when it is correctly specified, and predicts worse on realistic data. Collapsed fits are kept as ordinary fits and count as zero recovery.

## P1 PRO-ACT survival calibration (kyMv)

kyMv noted that the real-data evidence rests on held-out likelihood and asked for survival calibration. We refit the PRO-ACT models (10 seeds × d_z ∈ {0, 1}) on the paper's splits. They reproduce the paper's C-index, IBS-IPCW, best epoch and held-out likelihoods exactly. Every quantity was scored under the prior, the revision's predictor; q_agg gives the same numbers (previous sections). Two calibration diagnostics:
- **D-calibration:** checks whether each patient's predicted probability of surviving past their own observed time, S(tᵢ | xᵢ), is spread uniformly between 0 and 1, as it is for a well-calibrated model. The probabilities are grouped into 10 bins (SurvivalEVAL). The statistic measures how far the bin counts are from equal, so lower is better. A seed passes when p > 0.05, meaning calibration cannot be rejected.
- **KM-calibration:** measures how far the average predicted curve is from the test-set Kaplan–Meier curve.

| PRO-ACT, prior prediction | d_z = 0 | d_z = 1 | d_z = 1 better in |
|---|---|---|---|
| D-calibration statistic (mean ± SD) | 14.9 ± 5.5 | 8.0 ± 4.1 | 9 of 10 seeds |
| Seeds passing D-calibration (p > 0.05) | 6 of 10 | 9 of 10 | — |
| Max bin deviation | 0.032 | 0.020 | 9 of 10 seeds |
| KM-calibration | **0.0005** | 0.0059 | 0 of 10 seeds |
| Held-out event-margin log-likelihood | −4.268 | **−4.208** | 10 of 10 seeds |
| Held-out censoring-margin log-likelihood | −2.758 | **−2.558** | 10 of 10 seeds |
| Held-out joint log-likelihood (IWAE) | −7.026 | **−6.732** | 10 of 10 seeds |

**What we see.**
- **The latent gives better D-calibration: per-patient survival predictions are better calibrated.** The statistic drops from 14.9 to 8.0 (lower in 9 of 10 seeds), and the model passes in 9 of 10 seeds instead of 6. Without the latent, the D-calibration histogram is U-shaped: too many patients fall in the extreme bins (S near 0 or near 1). Its predictive distributions are therefore too narrow, or overconfident, and events happen earlier or later than predicted more often than they should. Averaging over the latent widens each patient's predictive distribution, which removes most of the U-shape.
- **The likelihood gain holds under the prior, in every seed and on both margins.** This includes the event margin, which is the likelihood of the reported survival curve itself, so the paper's real-data claim is unchanged by the switch.
- **On average, the latent-free curve sits closer to Kaplan–Meier.** KM-calibration compares only the population-average curve, not individual patients. Kaplan–Meier assumes independent censoring and is itself biased if censoring is dependent.
- **Both diagnostics assume independent censoring,** including D-calibration's handling of censored patients. They are descriptive.

As the paper already says, these results show a better fit to observed follow-up, not that censoring in PRO-ACT is dependent.

## Not on the list, but affects claims in the paper

**Fig. 6 latent recovery on all 12 datasets.** The abstract claims DVFM recovers subject-level frailty better than scalar baselines, mainly on censored subjects, but Fig. 6 shows only the three SEER cohorts. We extended it to all 12 datasets (τ = 0.5, 4 copulas, 10 seeds). DVFM's side comes from the refits of the prior vs aggregate-posterior comparison, which are identical to the paper's fits.

| Censored subjects, Δρ = DVFM − baseline | DVFM better | Tied | DVFM worse | Median Δρ (95% bootstrap CI) |
|---|---|---|---|---|
| CG posterior frailty | 11 | 1 | 0 | +0.19 [0.13, 0.28] |
| CoxPH martingale residual | 4 (SEER ×3, FLCHAIN) | 4 | 4 (GBSG, WHAS, CHURN, SUPPORT) | −0.00 [−0.01, 0.13] |

**What we see.** Against CG the claim generalises well: DVFM wins on 11 of 12 datasets. Against the CoxPH martingale residual it does not: DVFM wins on only 4 datasets, and those include the three SEER cohorts the paper happens to show. The claim should be restricted to CG, or both baselines shown across all datasets.

**Missing synthetic artifact.** The synthetic notebook reads `results/synthetic`, which does not exist on the server, so the synthetic figures in the paper (dated Sep 22) cannot be traced to any run. The other `results/synthetic-*` directories are older studies run under different settings. The synthetic figures should be regenerated from a fresh run before they are cited.

**Synthetic τ recovery.** Our new d_z = 1 synthetic fits, under the current protocol, recover τ well only near τ = 0.5 (mean absolute error 0.09). At τ = 0 the estimate is off by 0.36 on average, with either sign; at τ = 0.25 by 0.28; and at τ = 0.75 it is too low by 0.23. Any claim that DVFM recovers dependence strength across the grid is not supported.

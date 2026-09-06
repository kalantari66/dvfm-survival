# DVFM Latent-Variable Ablation: $d_z=20$ vs. $d_z=0$

## Overview

This report evaluates the contribution of the latent frailty variable in the Deep Variational Frailty Model (DVFM). We compare:

- **DVFM, $d_z=20$:** the complete model, trained with the C-ELBO and predicted using aggregate-posterior marginalization.
- **DVFM, $d_z=0$:** the original DVFM code with `latent_dim = 0` during both training and testing. In this case, the encoder returns empty latent tensors, the KL term is exactly zero, the decoder receives only $x$, and prediction is deterministic.

The comparison addresses this question:

Does the latent representation improve event-time prediction?

## Experimental setup

The prediction ablation used ten synthetic dependent-censoring scenarios: low/high Clayton, frailty-mixture, Frank, Gaussian, and Gumbel dependence. The matched configuration was:

- $N=5000$ observations per scenario
- 200 training epochs
- batch size 64
- learning rate $0.001$
- $\beta_{\max}=1$
- 100 Monte Carlo samples for $d_z=20$
- identical random seed and train/test split for both models


## Prediction results

The table reports the mean performance across the ten matched synthetic scenarios.

| Model | C-index $\uparrow$ | IBS $\downarrow$ | MAE $\downarrow$ | Oracle C-index $\uparrow$ | Oracle IBS $\downarrow$ | Oracle MAE $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|
| **DVFM, $d_z=20$** | **0.8204** | **0.0869** | **0.3405** | **0.7716** | **0.0639** | **1.1463** |
| DVFM, $d_z=0$ | 0.8150 | 0.0912 | 0.3686 | 0.7513 | 0.0721 | 1.2310 |

Relative to $d_z=0$, the latent DVFM achieved:

- $+0.0054$ C-index;
- $-0.0043$ IBS;
- $-0.0281$ MAE;
- $+0.0204$ oracle C-index;
- $-0.0081$ oracle IBS;
- $-0.0847$ oracle MAE.

The latent model won in 9/10 scenarios for C-index, 10/10 for IBS, 9/10 for MAE, 10/10 for oracle C-index, and 9/10 for both oracle IBS and oracle MAE.

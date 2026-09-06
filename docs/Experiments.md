### Experiment A: Clayton copula

Purpose:

- test robustness to a dependence mechanism not explicitly parameterized as Gaussian shared frailty;
- test event-distribution recovery under tail dependence;
- compare DVFM against a correctly specified Clayton model.

Known truth:

Ei, Ci, θ, τ, UE,i, UC,i.E_i,\ C_i,\ \theta,\ \tau,\ U_{E,i},\ U_{C,i}.Ei​, Ci​, θ, τ, UE,i​, UC,i​.

Not uniquely known:

Zi.Z_i.Zi​.

### Experiment B: one shared Gaussian frailty

Purpose:

- test the model under a correctly aligned inductive bias;
- directly evaluate whether the inferred latent tracks the generating ZiZ_iZi​.

Known truth:

Ei, Ci, Zi.E_i,\ C_i,\ Z_i.Ei​, Ci​, Zi​.

### Experiment C: shared and private frailties

Purpose:

- distinguish dependence recovery from marginal heterogeneity;
- compare the current single-zzz DVFM with the proposed zs,zE,zCz_s,z_E,z_Czs​,zE​,zC​ model.

Known truth:

Zs,i, ZE,i, ZC,i, Ei, Ci.Z_{s,i},\ Z_{E,i},\ Z_{C,i},\ E_i,\ C_i.Zs,i​, ZE,i​, ZC,i​, Ei​, Ci​.

### Experiment D: misspecified mechanism

Purpose:

- determine where DVFM fails;
- use a mechanism that cannot be well represented by one shared frailty, such as multimodal, sign-changing, or direct E→CE\to CE→C dependence.

This sequence directly supports the paper’s own caution that the full neural DVFM should be understood as a predictive inductive bias, not as guaranteeing recovery of the true joint law or latent frailty.

This is the distinction your experiments should make:

1. **Learning the population dependence mechanism**
2. **Inferring an individual latent frailty after observing follow-up**
3. **Baseline prediction from xxx alone**

The first two may work even when the third cannot exploit subject-specific ziz_izi​.



The main paper evidence should eventually come from:

1. a compact synthetic mechanistic section showing the method behaves correctly under controlled conditions;
2. a broad semi-synthetic section showing the effect survives realistic data;
3. real data, if useful, with appropriately modest claims because the true event times/dependence are unknown.
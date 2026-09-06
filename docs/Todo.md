- **Synthetic mechanistic benchmark**  
    Run the full grid dz​∈{0,1,5,20}, τ∈{0,.25,.5,.75}, censoring 30/50/70%, shared vs separate latent, 10 seeds.
    
- **Semi-synthetic robustness benchmark**  
    Use perhaps 5–10 datasets like SUPPORT, but with a much smaller set of settings, e.g.:
    - dz​∈{0,1,5,10,20}
    - τ∈{0,0.5,0.75}
    - Censoring=25, 50, 75
    - 10 seeds.

Experiment: synthetic_gaussian_latent

| Result                                                        | Next step                                   |
| ------------------------------------------------------------- | ------------------------------------------- |
| Shared > no-latent, shared > separate, learned (\tau) correct | **(2) Semi-synthetic 10 datasets**          |
| Only dz=20 works                                              | **(1) More targeted synthetic diagnostics** |
| Shared ≈ separate                                             | **(3) Refactor/inspect mechanism**          |
| Shared fails even in easy-mode DGP                            | **(3) Refactor objective/model**            |
| dz=1 works and mechanism checks pass                          | **(2) Semi-synthetic immediately**          |

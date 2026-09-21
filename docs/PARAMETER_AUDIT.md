# Reference parameter audit

The benchmark configuration uses the values in the active `run_experiment()` and function defaults of `reference/VAE_montcarlo.py`.

| Setting | Value used in this repository | Reference location |
|---|---:|---|
| Synthetic sample size | 10,000 | `VAE_montcarlo.py`, active `generate_copula_data(...)` call |
| Number of features | 10 | active `generate_copula_data(...)` call |
| Holdout test fraction | 0.30 | active `train_test_split(...)` call |
| Random seed | 42 | generator default and split |
| Time-grid points | 1,000 | active `np.linspace(...)` call |
| Maximum evaluation time | `1.5 × max(train time)` | active experiment |
| DVFM epochs | **200** | active `train_dvfm(..., n_epochs=200)` call |
| DVFM learning rate | 0.001 | `train_dvfm` default; active call does not override it |
| DVFM batch size | 64 | active DataLoaders |
| DVFM latent dimension | 20 | `Latent = 20` |
| KL beta maximum | 1.0 | active training call |
| KL warm-up epochs | 50 | `train_dvfm` default |
| Free bits | 0.0 | active training call |
| Monte Carlo samples | 100 | active aggregate-posterior prediction call |
| DeepSurv epochs | 200 | `train_deepsurv` default |
| DeepSurv learning rate | 0.001 | `train_deepsurv` default |
| DeepSurv batch size | 64 | `train_deepsurv` default |
| MTLR epochs | 200 | `train_mtlr` default |
| MTLR learning rate | 0.005 | `train_mtlr` default |
| MTLR bins in active experiment | 200 | active `train_mtlr(..., num_bins=200)` call |
| ClaytonAFT epochs | 100 | modified reference experiment configuration |
| ClaytonAFT learning rate | 0.005 | modified reference experiment configuration |
| Synthetic repetitions in paper config | 5 | manuscript and modified reference runner |

## Important

There is no performance-oriented `--quick` option in this revision. A shortened run can verify execution, but its model ranking is not a valid reproduction test. Use `--validate-only` for an installation/input check that does not alter any training parameter.

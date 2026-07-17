# Repository test report

Tested in a clean package import context on 2026-07-16.

## Automated checks

```text
Python compilation: passed
Package import: passed
Reference configuration validation: passed
pytest: 3 passed
```

The tests verify:

- exact DVFM reference settings, including 200 epochs;
- synthetic data shapes and event indicators;
- finite DVFM C-ELBO output;
- aggregate-posterior prediction dimensions;
- finite oracle and IPCW Brier scores.

## End-to-end interface checks

Both included data interfaces completed successfully with the full configured DVFM training length of 200 epochs:

```text
configs/real_example.yaml: completed and wrote all CSV outputs
configs/semi_synthetic_example.yaml: completed and wrote all CSV outputs
```

These toy datasets verify execution only. Their model rankings are not paper-reproduction results.

The full 10,000-sample, five-repeat synthetic benchmark was not run during packaging because it is computationally expensive. Its configuration was validated and is provided in `configs/paper_synthetic.yaml`.

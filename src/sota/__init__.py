"""State-of-the-art survival baselines used by DVFM experiments.

Models copied or adapted from external repositories keep their provenance in
``src/sota/README.md``. Overlapping baselines deliberately delegate to the
paper's preserved reference implementations.
"""

__all__ = [
    "adapters", "baselines", "clayton_aft", "coxph", "deephit",
    "deepsurv", "hacsurv", "mtlr", "sksurv", "weibull_aft",
]

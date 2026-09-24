"""State-of-the-art survival baselines used by DVFM experiments.

Models copied or adapted from external repositories keep their provenance in
their own module header. Overlapping baselines deliberately delegate to the
paper's preserved reference implementations.
"""

__all__ = [
    "adapters", "baselines", "bayesian_cox_gamma_frailty", "clayton_aft", "coxph",
    "deepsurv", "hacsurv", "mtlr", "sksurv", "weibull_aft",
]

"""expengine: a small experimentation library.

Power / sample size, SRM detection, CUPED, delta-method ratio metrics, mSPRT always-valid
inference, Beta-Binomial / Normal Bayesian analysis, segment effects with BH correction,
plus a SYNTHETIC experiment simulator, Monte Carlo studies and Markdown readouts.
"""

from expengine.bayes import bayes_decision, beta_binomial, normal_posterior
from expengine.cuped import EffectEstimate, cuped_estimate, cuped_theta, diff_in_means, regression_adjusted
from expengine.power import mde_means, n_per_arm_means, n_per_arm_proportions, power_means
from expengine.ratio import ratio_mean_var, ratio_metric_test, relative_lift
from expengine.sequential import always_valid_ci, msprt_lambda, msprt_pvalues, msprt_sequence
from expengine.srm import SRMResult, srm_test

__all__ = [
    "EffectEstimate",
    "SRMResult",
    "always_valid_ci",
    "bayes_decision",
    "beta_binomial",
    "cuped_estimate",
    "cuped_theta",
    "diff_in_means",
    "mde_means",
    "msprt_lambda",
    "msprt_pvalues",
    "msprt_sequence",
    "n_per_arm_means",
    "n_per_arm_proportions",
    "normal_posterior",
    "power_means",
    "ratio_mean_var",
    "ratio_metric_test",
    "regression_adjusted",
    "relative_lift",
    "srm_test",
]

"""Simulated discount A/B experiment with two-proportion z-test."""
import numpy as np
from scipy.stats import norm


def run_experiment(baseline_churn, discount_effect, n_users, seed=42):
    """
    Simulate two groups:
      - A (control)   : churn rate = baseline_churn
      - B (treatment) : churn rate = baseline_churn - discount_effect (clipped at 0)
    """
    rng = np.random.default_rng(seed)
    n_a = n_users // 2
    n_b = n_users - n_a
    p_a = float(baseline_churn)
    p_b = max(0.0, p_a - discount_effect)

    churns_a = rng.binomial(n_a, p_a)
    churns_b = rng.binomial(n_b, p_b)

    rate_a = churns_a / n_a
    rate_b = churns_b / n_b

    pooled = (churns_a + churns_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (rate_b - rate_a) / se if se > 0 else 0.0
    p_value = float(2 * (1 - norm.cdf(abs(z))))

    return {
        "n_a": n_a, "n_b": n_b,
        "churns_a": int(churns_a), "churns_b": int(churns_b),
        "rate_a": float(rate_a), "rate_b": float(rate_b),
        "p_a_theory": p_a, "p_b_theory": p_b,
        "absolute_lift": float(rate_a - rate_b),
        "relative_lift": float((rate_a - rate_b) / rate_a * 100) if rate_a else 0.0,
        "z_stat": float(z), "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "discount_effect": discount_effect,
    }
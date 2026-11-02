"""Random variable simulation (Binomial + continuous distributions)."""
import numpy as np
from scipy import stats


def binomial_simulation(n_trials, p_success, n_samples=1000, seed=42):
    """Simulate number of churners in `n_samples` groups of size `n_trials`."""
    rng = np.random.default_rng(seed)
    samples = rng.binomial(n=n_trials, p=p_success, size=n_samples)
    return {
        "samples": samples,
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)),
        "theoretical_mean": n_trials * p_success,
        "theoretical_std": float(np.sqrt(n_trials * p_success * (1 - p_success))),
        "n": n_trials,
        "p": p_success,
        "n_samples": n_samples,
    }


def normal_simulation(mu, sigma, n_samples=1000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n_samples)


def exponential_simulation(scale, n_samples=1000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=scale, size=n_samples)
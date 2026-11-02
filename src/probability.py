"""Sampling, conditional probability and Bayes' theorem."""
import numpy as np
import pandas as pd


def sample_estimates(df, col, n, n_samples=500, seed=42):
    """Sample `n_samples` samples of size `n` from `col`, return stats + means array."""
    rng = np.random.default_rng(seed)
    values = df[col].dropna().to_numpy()
    means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_samples)
    ])
    return {
        "population_mean": float(values.mean()),
        "population_std": float(values.std(ddof=1)),
        "sample_mean_of_means": float(means.mean()),
        "sample_variance_of_means": float(means.var(ddof=1)),
        "standard_error": float(values.std(ddof=1) / np.sqrt(n)),
        "means": means,
        "n": n,
        "n_samples": n_samples,
    }


def confidence_interval(values, confidence=0.95):
    """Normal-approximation confidence interval for the mean."""
    from scipy.stats import norm
    n = len(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    z = norm.ppf((1 + confidence) / 2)
    return {
        "mean": mean, "se": se, "z": z,
        "lower": mean - z * se, "upper": mean + z * se,
        "confidence": confidence, "n": n,
    }


def conditional_probability(df, group_col, group_val, target="churned"):
    """P(target | group_col == group_val)."""
    sub = df[df[group_col] == group_val]
    if len(sub) == 0:
        return {"group": group_val, "p": np.nan, "n": 0}
    return {"group": group_val, "p": float(sub[target].mean()), "n": int(len(sub))}


def marginal_probabilities(df, group_col, target="churned"):
    """P(target) and P(group_col = each value)."""
    p_target = float(df[target].mean())
    p_group = df[group_col].value_counts(normalize=True).round(4).to_dict()
    return {"p_target": p_target, "p_group": p_group}


def bayes_theorem(df, group_col, group_val, target="churned"):
    """
    Bayes:  P(target | group) = P(group | target) * P(target) / P(group)
    """
    n = len(df)
    p_target = df[target].mean()
    sub_target = df[df[target] == 1]
    sub_group = df[df[group_col] == group_val]

    p_group_given_target = float((sub_target[group_col] == group_val).mean())
    p_group = len(sub_group) / n
    p_target_given_group = (p_group_given_target * p_target) / p_group if p_group else np.nan

    return {
        "p_target": float(p_target),
        "p_group_given_target": p_group_given_target,
        "p_group": float(p_group),
        "p_target_given_group": float(p_target_given_group),
    }
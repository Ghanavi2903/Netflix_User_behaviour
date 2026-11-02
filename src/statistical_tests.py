"""Hypothesis tests: t-test and chi-square."""
import numpy as np
import pandas as pd
from scipy import stats


def two_sample_t_test(df, value_col, target="churned"):
    """Welch's t-test: watch time of churned vs non-churned."""
    g1 = df.loc[df[target] == 0, value_col].dropna().to_numpy()
    g2 = df.loc[df[target] == 1, value_col].dropna().to_numpy()
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    return {
        "n_non_churn": len(g1), "n_churn": len(g2),
        "mean_non_churn": float(g1.mean()), "mean_churn": float(g2.mean()),
        "std_non_churn": float(g1.std(ddof=1)), "std_churn": float(g2.std(ddof=1)),
        "t_statistic": float(t), "p_value": float(p),
        "significant": bool(p < 0.05),
    }


def chi_square_test(df, col, target="churned"):
    """Chi-square test of independence between `col` and `target`."""
    table = pd.crosstab(df[col], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    cramers_v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))) \
        if min(table.shape) > 1 else np.nan
    return {
        "table": table,
        "chi2": float(chi2), "p_value": float(p),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
        "significant": bool(p < 0.05),
    }
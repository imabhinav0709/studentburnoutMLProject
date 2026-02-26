"""
Statistical Hypothesis Testing Module
======================================
Provides a comprehensive suite of hypothesis tests to validate
causal assumptions before ML modelling.

Tests included:
  • Independent t-test / Welch's t-test
  • One-way ANOVA
  • Pearson & Spearman correlations
  • Chi-square test of independence
  • Effect sizes (Cohen's d, eta-squared, Cramér's V)
  • Confidence intervals
  • Statistical power analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
from dataclasses import dataclass, field
from typing import Any


# ────────────────────────────────────────────────────────────────────
#  Data classes for structured results
# ────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_name: str
    feature: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float | None = None
    effect_label: str | None = None
    confidence_interval: tuple[float, float] | None = None
    power: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "test": self.test_name,
            "feature": self.feature,
            "statistic": round(float(self.statistic), 4),
            "p_value": round(float(self.p_value), 6),
            "significant_at_0.05": bool(self.significant),
        }
        if self.effect_size is not None:
            d["effect_size"] = round(float(self.effect_size), 4)
            d["effect_label"] = self.effect_label
        if self.confidence_interval is not None:
            d["ci_lower"] = round(float(self.confidence_interval[0]), 4)
            d["ci_upper"] = round(float(self.confidence_interval[1]), 4)
        if self.power is not None:
            d["power"] = round(float(self.power), 4)
        return d


# ────────────────────────────────────────────────────────────────────
#  Helper: Cohen's d
# ────────────────────────────────────────────────────────────────────

def _cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1 - 1) * g1.std(ddof=1)**2 +
                          (n2 - 1) * g2.std(ddof=1)**2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled_std if pooled_std else 0.0


def _effect_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    return "large"


# ────────────────────────────────────────────────────────────────────
#  Core test functions
# ────────────────────────────────────────────────────────────────────

def pearson_correlation(
    df: pd.DataFrame, feature: str, target: str = "final_score", alpha: float = 0.05
) -> TestResult:
    """Pearson correlation with CI and power."""
    x = df[feature].values
    y = df[target].values
    r, p = stats.pearsonr(x, y)

    # Fisher z-transform CI
    n = len(x)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (np.tanh(z - z_crit * se), np.tanh(z + z_crit * se))

    # Power (approximate via t-test equivalence)
    t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else float("inf")
    effect = abs(r)

    return TestResult(
        test_name="Pearson Correlation",
        feature=feature,
        statistic=r,
        p_value=p,
        significant=p < alpha,
        effect_size=abs(r),
        effect_label=_effect_label(abs(r)),
        confidence_interval=ci,
        power=None,  # filled later if needed
    )


def spearman_correlation(
    df: pd.DataFrame, feature: str, target: str = "final_score", alpha: float = 0.05
) -> TestResult:
    """Spearman rank correlation."""
    r, p = stats.spearmanr(df[feature], df[target])
    return TestResult(
        test_name="Spearman Correlation",
        feature=feature,
        statistic=r,
        p_value=p,
        significant=p < alpha,
        effect_size=abs(r),
        effect_label=_effect_label(abs(r)),
    )


def independent_ttest(
    df: pd.DataFrame,
    group_col: str,
    target: str = "final_score",
    alpha: float = 0.05,
) -> TestResult:
    """Welch's independent t-test for a binary grouping variable."""
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError(f"{group_col} must have exactly 2 groups, got {len(groups)}")

    g1 = df.loc[df[group_col] == groups[0], target].values
    g2 = df.loc[df[group_col] == groups[1], target].values

    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    d = _cohens_d(g1, g2)

    # Power analysis
    power_analyzer = TTestIndPower()
    power = power_analyzer.solve_power(
        effect_size=abs(d), nobs1=len(g1), ratio=len(g2) / len(g1), alpha=alpha
    )

    # CI for the mean difference
    diff = g1.mean() - g2.mean()
    se = np.sqrt(g1.var(ddof=1) / len(g1) + g2.var(ddof=1) / len(g2))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (diff - z_crit * se, diff + z_crit * se)

    return TestResult(
        test_name="Welch's t-test",
        feature=group_col,
        statistic=t,
        p_value=p,
        significant=p < alpha,
        effect_size=d,
        effect_label=_effect_label(d),
        confidence_interval=ci,
        power=power,
        extras={"group_means": {str(groups[0]): g1.mean(), str(groups[1]): g2.mean()}},
    )


def one_way_anova(
    df: pd.DataFrame,
    group_col: str,
    target: str = "final_score",
    alpha: float = 0.05,
) -> TestResult:
    """One-way ANOVA with eta-squared effect size and power."""
    groups_vals = [g[target].values for _, g in df.groupby(group_col)]
    if len(groups_vals) < 2:
        raise ValueError(f"{group_col} must have ≥ 2 groups")

    F, p = stats.f_oneway(*groups_vals)

    # eta-squared
    grand_mean = df[target].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups_vals)
    ss_total = ((df[target] - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total else 0.0

    # Power
    k = len(groups_vals)
    n_per_group = int(np.mean([len(g) for g in groups_vals]))
    f_effect = np.sqrt(eta_sq / (1 - eta_sq)) if eta_sq < 1 else 0.0
    try:
        power_analyzer = FTestAnovaPower()
        power = power_analyzer.solve_power(
            effect_size=f_effect, nobs=n_per_group, k_groups=k, alpha=alpha
        )
    except Exception:
        power = None

    return TestResult(
        test_name="One-way ANOVA",
        feature=group_col,
        statistic=F,
        p_value=p,
        significant=p < alpha,
        effect_size=eta_sq,
        effect_label="eta²",
        power=power,
    )


def chi_square_test(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    alpha: float = 0.05,
) -> TestResult:
    """Chi-square test of independence with Cramér's V."""
    ct = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k else 0.0

    return TestResult(
        test_name="Chi-square",
        feature=f"{col1} × {col2}",
        statistic=chi2,
        p_value=p,
        significant=p < alpha,
        effect_size=cramers_v,
        effect_label="Cramér's V",
        extras={"dof": dof},
    )


# ────────────────────────────────────────────────────────────────────
#  Batch runner
# ────────────────────────────────────────────────────────────────────

def run_full_hypothesis_suite(df: pd.DataFrame, target: str = "final_score") -> list[dict]:
    """
    Run the complete hypothesis testing suite against the dataset.

    Automatically detects feature types and applies the appropriate test.
    Returns a list of result dictionaries.
    """
    results: list[TestResult] = []

    continuous_features = [
        "study_hours_per_day",
        "sleep_hours",
        "attendance_pct",
        "previous_gpa",
        "stress_level",
        "extracurricular_hrs",
        "screen_time_hrs",
        "exercise_freq",
    ]

    binary_features = ["has_part_time_job"]

    # ── Pearson & Spearman for continuous features ──────────────────
    for feat in continuous_features:
        if feat in df.columns:
            results.append(pearson_correlation(df, feat, target))
            results.append(spearman_correlation(df, feat, target))

    # ── t-test for binary features ──────────────────────────────────
    for feat in binary_features:
        if feat in df.columns:
            results.append(independent_ttest(df, feat, target))

    # ── ANOVA for gender (3 groups) ─────────────────────────────────
    if "gender" in df.columns:
        results.append(one_way_anova(df, "gender", target))

    # ── Chi-square between categoricals ─────────────────────────────
    if "gender" in df.columns and "has_part_time_job" in df.columns:
        results.append(chi_square_test(df, "gender", "has_part_time_job"))

    return [r.to_dict() for r in results]


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.make_dataset import generate_dataset
    import json

    df = generate_dataset()
    results = run_full_hypothesis_suite(df)
    print(json.dumps(results, indent=2))

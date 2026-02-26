"""
Model Interpretability Module
=============================
SHAP-based explanations + comparison with statistical hypothesis
testing results to demonstrate mature, engineer-level analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt

from src.models.train_models import load_best_model


# ────────────────────────────────────────────────────────────────────
#  SHAP Explanations
# ────────────────────────────────────────────────────────────────────

def compute_shap_values(
    X: pd.DataFrame,
    save_dir: str | Path = "models",
) -> dict:
    """
    Compute SHAP values using the best model.

    Returns
    -------
    dict with:
      - feature_importance : dict[str, float]   mean |SHAP| per feature
      - shap_values        : np.ndarray          raw SHAP matrix
      - expected_value     : float
    """
    pipeline, meta = load_best_model(save_dir)

    # Extract the fitted model and scaler from the pipeline
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    X_scaled = pd.DataFrame(
        scaler.transform(X), columns=X.columns, index=X.index
    )

    # Use TreeExplainer for tree models, KernelExplainer fallback
    model_type = type(model).__name__
    if model_type in ("RandomForestRegressor", "GradientBoostingRegressor", "XGBRegressor"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_scaled)
    else:
        # Linear / other — use a sample background
        bg = shap.sample(X_scaled, min(100, len(X_scaled)))
        explainer = shap.KernelExplainer(model.predict, bg)
        sv = explainer.shap_values(X_scaled)

    mean_abs = np.abs(sv).mean(axis=0)
    importance = dict(zip(X.columns, mean_abs.round(4)))
    importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    return {
        "feature_importance": importance,
        "shap_values": sv,
        "expected_value": float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0]),
        "model_name": meta["best_model_name"],
    }


# ────────────────────────────────────────────────────────────────────
#  SHAP Visualisations
# ────────────────────────────────────────────────────────────────────

def save_shap_plots(
    shap_result: dict,
    X: pd.DataFrame,
    output_dir: str | Path = "outputs/plots",
):
    """Generate and save SHAP summary + bar plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sv = shap_result["shap_values"]

    # ── Bar plot (mean |SHAP|) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    imp = shap_result["feature_importance"]
    features = list(imp.keys())
    values = list(imp.values())
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    ax.barh(features[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance — {shap_result['model_name']}")
    plt.tight_layout()
    fig.savefig(out / "shap_bar.png", dpi=150)
    plt.close(fig)

    # ── Beeswarm / summary plot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=150)
    plt.close("all")

    print(f"[OK] SHAP plots saved -> {out.resolve()}")


# ────────────────────────────────────────────────────────────────────
#  Statistical vs ML Comparison
# ────────────────────────────────────────────────────────────────────

def build_comparison_table(
    stat_results: list[dict],
    shap_importance: dict[str, float],
) -> pd.DataFrame:
    """
    Merge hypothesis testing p-values with SHAP feature importance
    to produce a side-by-side comparison table.
    """
    # Take the Pearson correlation results (one per continuous feature)
    stat_rows = {}
    for r in stat_results:
        if r["test"] == "Pearson Correlation":
            feat = r["feature"]
            stat_rows[feat] = {
                "p_value": r["p_value"],
                "statistically_significant": r["significant_at_0.05"],
                "pearson_r": r["statistic"],
            }
        elif r["test"] == "Welch's t-test":
            feat = r["feature"]
            stat_rows[feat] = {
                "p_value": r["p_value"],
                "statistically_significant": r["significant_at_0.05"],
                "pearson_r": None,
            }

    rows = []
    for feat, shap_val in shap_importance.items():
        row = {"feature": feat, "shap_importance": shap_val}
        if feat in stat_rows:
            row.update(stat_rows[feat])
        else:
            row["p_value"] = None
            row["statistically_significant"] = None
            row["pearson_r"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("shap_importance", ascending=False).reset_index(drop=True)
    df["rank_shap"] = range(1, len(df) + 1)

    return df


def save_comparison(
    comparison_df: pd.DataFrame,
    output_dir: str | Path = "outputs",
):
    """Save comparison table as CSV and a styled plot."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out / "stat_vs_ml_comparison.csv", index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    features = comparison_df["feature"].tolist()
    x = np.arange(len(features))
    width = 0.35

    shap_vals = comparison_df["shap_importance"].values
    p_vals = comparison_df["p_value"].fillna(1.0).values
    neg_log_p = -np.log10(p_vals + 1e-300)

    # Normalise for visual comparison
    shap_norm = shap_vals / shap_vals.max() if shap_vals.max() else shap_vals
    p_norm = neg_log_p / neg_log_p.max() if neg_log_p.max() else neg_log_p

    ax1.bar(x - width / 2, shap_norm, width, label="SHAP importance (norm)", color="#6366f1")
    ax1.bar(x + width / 2, p_norm, width, label="-log₁₀(p-value) (norm)", color="#f59e0b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Normalised Value")
    ax1.set_title("Statistical Significance vs ML Feature Importance")
    ax1.legend()
    plt.tight_layout()
    fig.savefig(out / "plots" / "stat_vs_ml_comparison.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Comparison saved -> {out.resolve()}")


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    from src.data.make_dataset import generate_dataset, preprocess
    from src.features.statistics import run_full_hypothesis_suite

    df = generate_dataset()
    X, y = preprocess(df)

    # SHAP
    shap_result = compute_shap_values(X)
    save_shap_plots(shap_result, X)

    # Comparison
    stat_results = run_full_hypothesis_suite(df)
    comp = build_comparison_table(stat_results, shap_result["feature_importance"])
    save_comparison(comp)
    print(comp.to_string(index=False))

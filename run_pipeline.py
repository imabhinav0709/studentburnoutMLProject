"""
Pipeline Runner - Causal Student Performance Intelligence System
=================================================================
Runs the full pipeline end-to-end:
  1. Generate synthetic dataset
  2. Run hypothesis testing suite
  3. Train & evaluate ML models
  4. Compute SHAP explainability
  5. Build stat-vs-ML comparison

After this script finishes, start the backend + dashboard:
  python -m uvicorn src.api.main:app --reload --port 8000
  cd frontend && npm run dev
"""

import os
import sys
import json
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.make_dataset import generate_dataset, preprocess, save_dataset
from src.features.statistics import run_full_hypothesis_suite
from src.models.train_models import train_and_evaluate
from src.models.interpretability import (
    compute_shap_values,
    save_shap_plots,
    build_comparison_table,
    save_comparison,
)


def main():
    print("=" * 60)
    print("  Causal Student Performance Intelligence System")
    print("  Full Pipeline Execution")
    print("=" * 60)

    # -- Step 1: Data --
    print("\n[1/5] Generating synthetic dataset...")
    df = generate_dataset()
    save_dataset(df, "data/student_performance.csv")

    # -- Step 2: Hypothesis Testing --
    print("\n[2/5] Running hypothesis tests...")
    stat_results = run_full_hypothesis_suite(df)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hypothesis_results.json", "w") as f:
        json.dump(stat_results, f, indent=2)
    sig = sum(1 for r in stat_results if r["significant_at_0.05"])
    print(f"   -> {sig}/{len(stat_results)} tests significant at alpha=0.05")

    # -- Step 3: ML Training --
    print("\n[3/5] Training & tuning ML models...")
    X, y = preprocess(df)
    meta = train_and_evaluate(X, y, save_dir="models")

    # -- Step 4: SHAP Interpretability --
    print("\n[4/5] Computing SHAP values...")
    shap_result = compute_shap_values(X, "models")
    save_shap_plots(shap_result, X, "outputs/plots")

    # -- Step 5: Comparison --
    print("\n[5/5] Building stat vs ML comparison...")
    comp_df = build_comparison_table(stat_results, shap_result["feature_importance"])
    save_comparison(comp_df, "outputs")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("   1. Start API:        python -m uvicorn src.api.main:app --reload --port 8000")
    print("   2. Start Dashboard:  cd frontend && npm run dev")
    print()


if __name__ == "__main__":
    main()

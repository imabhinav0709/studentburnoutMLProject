"""
FastAPI Backend — Causal Student Performance Intelligence System
================================================================
Endpoints:
  POST /predict       – predict final score for a student
  GET  /statistics    – return hypothesis testing results
  GET  /comparison    – return stat vs ML comparison
  GET  /models        – return model evaluation metrics
  GET  /health        – health-check
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    StudentInput,
    PredictionResponse,
    StatisticsResponse,
    ComparisonResponse,
)
from src.data.make_dataset import generate_dataset
from src.features.statistics import run_full_hypothesis_suite
from src.models.train_models import load_best_model
from src.models.interpretability import compute_shap_values, build_comparison_table
from src.data.make_dataset import preprocess

# ── App init ────────────────────────────────────────────────────────
app = FastAPI(
    title="Causal Student Performance Intelligence System",
    version="1.0.0",
    description="AI/ML-powered student performance prediction with causal analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state (loaded once) ──────────────────────────────────────
_state: dict = {}


def _ensure_loaded():
    """Lazy-load model, data, and pre-computed results."""
    if _state:
        return

    import os
    os.chdir(PROJECT_ROOT)

    # Data
    df = generate_dataset()
    X, y = preprocess(df)

    # Model
    pipeline, meta = load_best_model("models")

    # Stats
    stat_results = run_full_hypothesis_suite(df)

    # SHAP
    shap_result = compute_shap_values(X, "models")

    # Comparison
    comp_df = build_comparison_table(stat_results, shap_result["feature_importance"])

    _state.update({
        "pipeline": pipeline,
        "meta": meta,
        "feature_names": meta["feature_names"],
        "stat_results": stat_results,
        "shap_result": shap_result,
        "comparison_df": comp_df,
        "X": X,
        "y": y,
    })


# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput):
    _ensure_loaded()
    pipeline = _state["pipeline"]
    feature_names = _state["feature_names"]
    shap_result = _state["shap_result"]

    # Build input DataFrame matching training feature order
    row = {
        "study_hours_per_day": student.study_hours_per_day,
        "sleep_hours": student.sleep_hours,
        "attendance_pct": student.attendance_pct,
        "previous_gpa": student.previous_gpa,
        "stress_level": student.stress_level,
        "extracurricular_hrs": student.extracurricular_hrs,
        "screen_time_hrs": student.screen_time_hrs,
        "exercise_freq": student.exercise_freq,
        "has_part_time_job": student.has_part_time_job,
    }

    # One-hot encode gender (match training: drop_first → Male dropped if alphabetical)
    for col in feature_names:
        if col.startswith("gender_"):
            suffix = col.replace("gender_", "")
            row[col] = 1 if student.gender == suffix else 0

    input_df = pd.DataFrame([row])[feature_names]
    pred = float(pipeline.predict(input_df)[0])
    pred = round(np.clip(pred, 0, 100), 2)

    # Simple confidence interval (±2 × training RMSE)
    best_result = next(
        r for r in _state["meta"]["results"]
        if r["model"] == _state["meta"]["best_model_name"]
    )
    rmse = best_result["test_rmse"]
    ci = [round(max(pred - 2 * rmse, 0), 2), round(min(pred + 2 * rmse, 100), 2)]

    # Top factors from SHAP
    imp = shap_result["feature_importance"]
    top = [{"feature": k, "importance": v} for k, v in list(imp.items())[:5]]

    return PredictionResponse(
        predicted_score=pred,
        confidence_interval=ci,
        top_factors=top,
        model_used=_state["meta"]["best_model_name"],
    )


@app.get("/statistics", response_model=StatisticsResponse)
def statistics():
    _ensure_loaded()
    return StatisticsResponse(results=_state["stat_results"])


@app.get("/comparison", response_model=ComparisonResponse)
def comparison():
    _ensure_loaded()
    comp = _state["comparison_df"].fillna("N/A").to_dict(orient="records")
    return ComparisonResponse(
        comparison=comp,
        model_used=_state["meta"]["best_model_name"],
    )


@app.get("/models")
def models():
    _ensure_loaded()
    return {"results": _state["meta"]["results"], "best": _state["meta"]["best_model_name"]}


# ── Run directly ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

"""
ML Training Pipeline
====================
Trains, tunes, and evaluates multiple regression models on the
student performance dataset.

Models:
  • Linear Regression
  • Random Forest Regressor
  • XGBoost Regressor
  • Gradient Boosting Regressor

Includes cross-validation, hyperparameter tuning via GridSearchCV,
and metric computation (RMSE, MAE, R²).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────────────────────────
#  Model Definitions & Hyperparameter Grids
# ────────────────────────────────────────────────────────────────────

MODEL_CONFIGS: dict[str, dict] = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {},  # no tuning needed
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [8, 12, None],
            "model__min_samples_split": [2, 5],
        },
    },
    "XGBoost": {
        "model": XGBRegressor(
            random_state=42, verbosity=0, n_jobs=-1
        ),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        },
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
        },
    },
}


# ────────────────────────────────────────────────────────────────────
#  Core Training Logic
# ────────────────────────────────────────────────────────────────────

def _build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    cv_folds: int = 5,
    save_dir: str | Path = "models",
) -> dict:
    """
    Train all models with hyperparameter tuning and cross-validation.

    Returns
    -------
    dict with keys:
      - results : list[dict]  per-model metrics
      - best_model_name : str
      - feature_names : list[str]
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    results = []
    best_score = -np.inf
    best_name = ""

    for name, cfg in MODEL_CONFIGS.items():
        print(f"[TRAIN] Training {name}...")
        pipe = _build_pipeline(cfg["model"])

        if cfg["params"]:
            grid = GridSearchCV(
                pipe,
                cfg["params"],
                cv=cv_folds,
                scoring="r2",
                n_jobs=-1,
                error_score="raise",
            )
            grid.fit(X_train, y_train)
            best_pipe = grid.best_estimator_
            best_params = grid.best_params_
        else:
            pipe.fit(X_train, y_train)
            best_pipe = pipe
            best_params = {}

        # ── Cross-validation on full training set ───────────────────
        cv_scores = cross_validate(
            best_pipe, X_train, y_train,
            cv=cv_folds,
            scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
            return_train_score=False,
        )

        # ── Test-set evaluation ─────────────────────────────────────
        y_pred = best_pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        result = {
            "model": name,
            "test_rmse": round(rmse, 4),
            "test_mae": round(mae, 4),
            "test_r2": round(r2, 4),
            "cv_r2_mean": round(float(cv_scores["test_r2"].mean()), 4),
            "cv_r2_std": round(float(cv_scores["test_r2"].std()), 4),
            "cv_rmse_mean": round(float(np.sqrt(-cv_scores["test_neg_mean_squared_error"].mean())), 4),
            "best_params": {k.replace("model__", ""): v for k, v in best_params.items()},
        }
        results.append(result)
        print(f"   [OK] R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

        # ── Track best ──────────────────────────────────────────────
        if r2 > best_score:
            best_score = r2
            best_name = name

        # ── Save model ──────────────────────────────────────────────
        model_file = save_path / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(best_pipe, model_file)

    # ── Save metadata ───────────────────────────────────────────────
    meta = {
        "results": results,
        "best_model_name": best_name,
        "feature_names": list(X.columns),
    }
    with open(save_path / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[BEST] Best model: {best_name} (R2={best_score:.4f})")
    return meta


# ────────────────────────────────────────────────────────────────────
#  Utility: load best model
# ────────────────────────────────────────────────────────────────────

def load_best_model(save_dir: str | Path = "models") -> tuple[Pipeline, dict]:
    """Load the best-performing model and its metadata."""
    save_path = Path(save_dir)
    with open(save_path / "training_meta.json") as f:
        meta = json.load(f)
    best_name = meta["best_model_name"]
    model_file = save_path / f"{best_name.lower().replace(' ', '_')}.joblib"
    pipeline = joblib.load(model_file)
    return pipeline, meta


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    from src.data.make_dataset import generate_dataset, preprocess

    df = generate_dataset()
    X, y = preprocess(df)
    meta = train_and_evaluate(X, y)
    print(json.dumps(meta["results"], indent=2))

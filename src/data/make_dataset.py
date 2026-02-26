"""
Synthetic Student Performance Dataset Generator
================================================
Generates a causally-structured dataset where features have known
relationships to the target (final_score), enabling ground-truth
validation of statistical tests and ML feature importance.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_STUDENTS = 2000

# ── Causal coefficients (ground truth) ──────────────────────────────
CAUSAL_WEIGHTS = {
    "study_hours_per_day":  4.5,   # strong positive
    "sleep_hours":          2.0,   # moderate positive
    "attendance_pct":       0.3,   # positive
    "previous_gpa":         8.0,   # strong positive
    "stress_level":        -3.0,   # negative
    "extracurricular_hrs":  0.5,   # weak positive
    "screen_time_hrs":     -1.5,   # moderate negative
    "exercise_freq":        1.0,   # weak positive
}

FEATURE_CONFIGS = {
    # feature_name: (mean, std, clip_low, clip_high)
    "study_hours_per_day":  (4.0, 2.0, 0.0, 14.0),
    "sleep_hours":          (7.0, 1.5, 3.0, 12.0),
    "attendance_pct":       (75.0, 15.0, 0.0, 100.0),
    "previous_gpa":         (3.0, 0.5, 0.0, 4.0),
    "stress_level":         (5.0, 2.0, 1.0, 10.0),
    "extracurricular_hrs":  (3.0, 2.0, 0.0, 10.0),
    "screen_time_hrs":      (4.0, 2.0, 0.0, 12.0),
    "exercise_freq":        (3.0, 1.5, 0.0, 7.0),
}

# categorical
GENDER_OPTIONS = ["Male", "Female", "Other"]
GENDER_PROBS   = [0.48, 0.48, 0.04]

PART_TIME_JOB_PROB = 0.30


def generate_dataset(n: int = N_STUDENTS, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic student performance dataset with causal structure."""
    rng = np.random.default_rng(seed)

    # ── Continuous features ─────────────────────────────────────────
    data = {}
    for feat, (mu, sigma, lo, hi) in FEATURE_CONFIGS.items():
        raw = rng.normal(mu, sigma, n)
        data[feat] = np.clip(raw, lo, hi)

    # ── Categorical features ────────────────────────────────────────
    data["gender"] = rng.choice(GENDER_OPTIONS, n, p=GENDER_PROBS)
    data["has_part_time_job"] = rng.binomial(1, PART_TIME_JOB_PROB, n)

    df = pd.DataFrame(data)

    # ── Target: final_score (0–100) ─────────────────────────────────
    score = np.full(n, 20.0)  # base score
    for feat, coeff in CAUSAL_WEIGHTS.items():
        score += coeff * df[feat].values

    # Add part-time job effect
    score -= 3.0 * df["has_part_time_job"].values

    # Add noise
    noise = rng.normal(0, 5, n)
    score += noise

    df["final_score"] = np.clip(score, 0, 100).round(2)

    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean and encode the dataset for modelling.

    Returns
    -------
    X : pd.DataFrame  – features (numeric, encoded)
    y : pd.Series      – target
    """
    df = df.copy()

    # One-hot encode gender
    df = pd.get_dummies(df, columns=["gender"], drop_first=True, dtype=int)

    y = df.pop("final_score")
    X = df

    return X, y


def save_dataset(df: pd.DataFrame, path: str | Path = "data/student_performance.csv"):
    """Save the dataset to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"[OK] Dataset saved -> {p.resolve()}  ({len(df)} rows, {df.shape[1]} cols)")


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    # Ensure project root is on path
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)

    df = generate_dataset()
    save_dataset(df)
    print(df.describe().round(2))

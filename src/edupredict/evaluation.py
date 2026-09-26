"""
EduPredict — model evaluation utilities.

Provides helpers to compute standard regression metrics and display a
comparison summary, mirroring the evaluation cells from training.ipynb.

Usage:
    from edupredict.evaluation import evaluate_models, print_report, score_pipeline

    report = evaluate_models(candidates, X_test, y_test)
    print_report(report)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a dict with R², RMSE, and MAE for a single set of predictions."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def score_pipeline(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Predict with *pipeline* on *X_test* and return metric dict."""
    y_pred = pipeline.predict(X_test)
    return compute_metrics(y_test, y_pred)


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------
def evaluate_models(
    candidates: dict[str, GridSearchCV],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate every GridSearchCV in *candidates* and return a summary DataFrame.

    Parameters
    ----------
    candidates : dict[str, GridSearchCV]
        Mapping of model name → fitted GridSearchCV (as produced by ``training.train``).
    X_test : pd.DataFrame
        Held-out feature matrix.
    y_test : pd.Series
        Held-out target vector.

    Returns
    -------
    pd.DataFrame
        Columns: model, cv_r2, test_r2, test_rmse, test_mae, best_params
    """
    rows = []
    for name, grid in candidates.items():
        y_pred = grid.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        rows.append(
            {
                "model": name,
                "cv_r2": round(grid.best_score_, 4),
                "test_r2": round(metrics["r2"], 4),
                "test_rmse": round(metrics["rmse"], 4),
                "test_mae": round(metrics["mae"], 4),
                "best_params": grid.best_params_,
            }
        )
        logger.info(
            "%s | CV R²=%.4f | test R²=%.4f | RMSE=%.4f | MAE=%.4f",
            name, grid.best_score_, metrics["r2"], metrics["rmse"], metrics["mae"],
        )

    report = pd.DataFrame(rows).sort_values("cv_r2", ascending=False).reset_index(drop=True)
    return report


def print_report(report: pd.DataFrame) -> None:
    """Pretty-print the evaluation report to stdout."""
    print("\n=== Model Evaluation Report ===")
    display_cols = ["model", "cv_r2", "test_r2", "test_rmse", "test_mae"]
    print(report[display_cols].to_string(index=False))
    best = report.iloc[0]
    print(f"\n✅ Best model: {best['model']}  (CV R² = {best['cv_r2']})")
    print("================================\n")

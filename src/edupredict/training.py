"""
EduPredict — model training pipeline.

Trains three regression models (LinearRegression, RandomForestRegressor, SVR)
with GridSearchCV + KFold cross-validation, picks the best, and persists the
pipeline + expected column list.

Steps (matching training.ipynb):
    1. Load processed CSV.
    2. Split into X / y and train/test sets.
    3. Grid-search LR, RF, SVR pipelines.
    4. Evaluate on the held-out test set.
    5. Save the best pipeline and expected column list via joblib.

Usage:
    from edupredict.training import train, load_processed

    X_train, X_test, y_train, y_test, feature_cols = load_processed(
        "data/processed/edu-predict-clean.csv"
    )
    best_pipeline = train(X_train, y_train)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirrors the notebook)
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TARGET_COL: str = "Exam_Score"
TEST_SIZE: float = 0.2
N_SPLITS: int = 3

# Default output paths (relative to project root)
DEFAULT_MODEL_PATH: str = "model/model_pipeline.pkl"
DEFAULT_COLUMNS_PATH: str = "model/expected_columns.pkl"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_processed(
    csv_path: str | Path = "data/processed/edu-predict-clean.csv",
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load the processed CSV and return (X_train, X_test, y_train, y_test, feature_cols)."""
    path = Path(csv_path)
    logger.info("Loading processed data from %s", path)
    df = pd.read_csv(path, index_col=0)
    # Drop any residual index column that may have been written by older pipelines
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    feature_cols = list(X_train.columns)
    logger.info(
        "Split: %d train / %d test rows, %d features",
        len(X_train), len(X_test), len(feature_cols),
    )
    return X_train, X_test, y_train, y_test, feature_cols


# ---------------------------------------------------------------------------
# Grid-search definitions
# ---------------------------------------------------------------------------
def _lr_grid(kf: KFold) -> GridSearchCV:
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    return GridSearchCV(
        pipe,
        {
            "linearregression__fit_intercept": [True, False],
            "linearregression__positive": [True, False],
        },
        cv=kf,
        scoring="r2",
        n_jobs=-1,
    )


def _rf_grid(kf: KFold) -> GridSearchCV:
    pipe = make_pipeline(
        StandardScaler(), RandomForestRegressor(random_state=RANDOM_STATE)
    )
    return GridSearchCV(
        pipe,
        {
            "randomforestregressor__n_estimators": [100, 200, 400],
            "randomforestregressor__max_depth": [None, 5, 10, 20],
            "randomforestregressor__min_samples_split": [2, 5, 10],
        },
        cv=kf,
        scoring="r2",
        n_jobs=-1,
    )


def _svr_grid(kf: KFold) -> GridSearchCV:
    pipe = make_pipeline(StandardScaler(), SVR())
    return GridSearchCV(
        pipe,
        {
            "svr__kernel": ["rbf", "linear", "poly"],
            "svr__C": [0.1, 1, 10, 100],
            "svr__epsilon": [0.01, 0.1, 0.2, 0.5],
        },
        cv=kf,
        scoring="r2",
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> Any:
    """
    Fit LR, RF, and SVR grid searches and return the best estimator pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    n_splits : int
        Number of KFold splits.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    best_pipeline
        The best sklearn Pipeline (StandardScaler + best model).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    candidates: dict[str, GridSearchCV] = {
        "LR": _lr_grid(kf),
        "RF": _rf_grid(kf),
        "SVR": _svr_grid(kf),
    }

    for name, grid in candidates.items():
        logger.info("Fitting %s …", name)
        grid.fit(X_train, y_train)
        logger.info("%s best CV R²: %.4f | params: %s", name, grid.best_score_, grid.best_params_)

    best_name = max(candidates, key=lambda n: candidates[n].best_score_)
    best_pipeline = candidates[best_name].best_estimator_
    logger.info("Best model: %s (CV R²=%.4f)", best_name, candidates[best_name].best_score_)
    return best_pipeline, candidates


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_artifacts(
    pipeline: Any,
    feature_cols: list[str],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    columns_path: str | Path = DEFAULT_COLUMNS_PATH,
) -> None:
    """Save the trained pipeline and column list with joblib."""
    for p in [Path(model_path), Path(columns_path)]:
        p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    joblib.dump(feature_cols, columns_path)
    logger.info("Saved pipeline → %s", model_path)
    logger.info("Saved columns  → %s", columns_path)


def load_artifacts(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    columns_path: str | Path = DEFAULT_COLUMNS_PATH,
) -> tuple[Any, list[str]]:
    """Load a previously saved pipeline and expected column list."""
    pipeline = joblib.load(model_path)
    feature_cols = joblib.load(columns_path)
    logger.info("Loaded pipeline from %s", model_path)
    return pipeline, feature_cols


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------
def run(
    processed_path: str | Path = "data/processed/edu-predict-clean.csv",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    columns_path: str | Path = DEFAULT_COLUMNS_PATH,
) -> None:
    """End-to-end: load data → train → evaluate → save."""
    X_train, X_test, y_train, y_test, feature_cols = load_processed(processed_path)
    best_pipeline, candidates = train(X_train, y_train)

    # Evaluate all candidates on test set
    for name, grid in candidates.items():
        y_pred = grid.predict(X_test)
        logger.info(
            "%s | best CV R²: %.4f | test R²: %.4f",
            name, grid.best_score_, r2_score(y_test, y_pred),
        )

    save_artifacts(best_pipeline, feature_cols, model_path, columns_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()

"""
EduPredict — preprocessing pipeline.

Converts the raw CSV into a clean, encoded DataFrame ready for model training.

Steps (matching cleaning.ipynb):
    1. Impute missing values in affected columns with their modes.
    2. Drop duplicate rows.
    3. Remove Exam_Score outliers via IQR.
    4. One-hot encode nominal columns (drop_first=False).
    5. Ordinal encode ordinal columns.
    6. Save processed CSV to data/processed/.

Usage:
    from edupredict.preprocessing import clean_data, load_raw, save_processed

    df_raw = load_raw("data/raw/dataset-6ab002b01f9ef078223946.csv")
    df_clean = clean_data(df_raw)
    save_processed(df_clean, "data/processed/edu-predict-clean.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column metadata (mirrors the notebook constants)
# ---------------------------------------------------------------------------
IMPUTE_COLS: list[str] = [
    "Teacher_Quality",
    "Parental_Education_Level",
    "Distance_from_Home",
]

NOMINAL_COLS: list[str] = [
    "Gender",
    "School_Type",
    "Extracurricular_Activities",
    "Internet_Access",
    "Learning_Disabilities",
    "Peer_Influence",
]

ORDINAL_COLS: list[str] = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level",
    "Family_Income",
    "Teacher_Quality",
    "Parental_Education_Level",
    "Distance_from_Home",
]

ORDINAL_CATEGORIES: list[list[str]] = [
    ["Low", "Medium", "High"],       # Parental_Involvement
    ["Low", "Medium", "High"],       # Access_to_Resources
    ["Low", "Medium", "High"],       # Motivation_Level
    ["Low", "Medium", "High"],       # Family_Income
    ["Low", "Medium", "High"],       # Teacher_Quality
    ["High School", "College", "Postgraduate"],  # Parental_Education_Level
    ["Near", "Moderate", "Far"],     # Distance_from_Home
]

TARGET_COL: str = "Exam_Score"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw dataset from *csv_path*."""
    path = Path(csv_path)
    logger.info("Loading raw data from %s", path)
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, csv_path: str | Path) -> None:
    """Persist the cleaned DataFrame to *csv_path* (creates parent dirs if needed)."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving processed data to %s (%d rows, %d cols)", path, *df.shape)
    df.to_csv(path, index=True)


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------
def impute_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1 & 2: fill missing values in affected columns; drop duplicates."""
    df = df.copy()
    for col in IMPUTE_COLS:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.debug("Imputed '%s' with mode '%s'", col, mode_val)
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", before - len(df))
    return df


def remove_outliers_iqr(df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
    """Step 3: remove rows where *col* falls outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    before = len(df)
    df = df[(df[col] >= lower) & (df[col] <= upper)].reset_index(drop=True)
    logger.info(
        "IQR outlier removal on '%s': bounds=[%.2f, %.2f], dropped %d rows",
        col, lower, upper, before - len(df),
    )
    return df


def encode_nominal(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Step 4: one-hot encode nominal columns (drop_first=False)."""
    if cols is None:
        cols = NOMINAL_COLS
    df = df.copy()
    # Normalise string casing before encoding
    for col in cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
    df = pd.get_dummies(df, columns=[c for c in cols if c in df.columns], drop_first=False)
    logger.info("One-hot encoded columns: %s", cols)
    return df


def encode_ordinal(df: pd.DataFrame, cols: list[str] | None = None,
                   categories: list[list[str]] | None = None) -> pd.DataFrame:
    """Step 5: ordinal encode *cols* using *categories* order."""
    if cols is None:
        cols = ORDINAL_COLS
    if categories is None:
        categories = ORDINAL_CATEGORIES
    df = df.copy()
    present = [c for c in cols if c in df.columns]
    cat_present = [categories[cols.index(c)] for c in present]
    if present:
        encoder = OrdinalEncoder(categories=cat_present)
        df[present] = encoder.fit_transform(df[present])
        logger.info("Ordinal encoded columns: %s", present)
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in order and return the processed DataFrame."""
    df = impute_and_deduplicate(df)
    df = remove_outliers_iqr(df)
    df = encode_nominal(df)
    df = encode_ordinal(df)
    return df


# ---------------------------------------------------------------------------
# Convenience runner (python -m edupredict.preprocessing)
# ---------------------------------------------------------------------------
def run(
    raw_path: str | Path = "data/raw/dataset-6ab002b01f9ef078223946.csv",
    output_path: str | Path = "data/processed/edu-predict-clean.csv",
) -> pd.DataFrame:
    """Load, clean, and save the dataset. Returns the cleaned DataFrame."""
    df_raw = load_raw(raw_path)
    df_clean = clean_data(df_raw)
    save_processed(df_clean, output_path)
    return df_clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()

"""Load, validate and clean product review data from CSV.

The loader never writes files. If no usable CSV is supplied, the bundled
synthetic sample (data/sample_reviews.csv) is used and clearly labelled.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

REQUIRED_COLUMNS = ["review_id", "product_name", "rating", "review_text", "review_date"]

# Alternative column names we accept and rename to the standard schema.
COLUMN_ALIASES = {
    "id": "review_id",
    "reviewid": "review_id",
    "product": "product_name",
    "productname": "product_name",
    "stars": "rating",
    "score": "rating",
    "text": "review_text",
    "review": "review_text",
    "reviewtext": "review_text",
    "comment": "review_text",
    "date": "review_date",
    "reviewdate": "review_date",
    "timestamp": "review_date",
}

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_CSV_PATH = DATA_DIR / "sample_reviews.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names, trim spaces, and map known aliases to the standard schema."""
    df = df.copy()
    renamed = {}
    for col in df.columns:
        clean = str(col).strip().lower().replace(" ", "_")
        if clean not in REQUIRED_COLUMNS:
            clean = COLUMN_ALIASES.get(clean.replace("_", ""), clean)
        renamed[col] = clean
    df = df.rename(columns=renamed)
    # If two source columns map to the same name, keep the first.
    return df.loc[:, ~df.columns.duplicated()]


def validate_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep usable rows and return (clean_dataframe, warnings).

    Raises ValueError if required columns are missing or no valid rows remain.
    """
    warnings: list[str] = []
    df = normalize_columns(df)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(REQUIRED_COLUMNS)}."
        )

    df = df[REQUIRED_COLUMNS].dropna(how="all").copy()

    for col in ("review_id", "product_name", "review_text"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    empty_text = df["review_text"] == ""
    if empty_text.any():
        warnings.append(f"Removed {int(empty_text.sum())} row(s) with empty review text.")
        df = df[~empty_text]

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    bad_rating = df["rating"].isna() | (df["rating"] < 1) | (df["rating"] > 5)
    if bad_rating.any():
        warnings.append(
            f"Removed {int(bad_rating.sum())} row(s) with a missing or out-of-range rating (must be 1-5)."
        )
        df = df[~bad_rating]

    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    bad_date = df["review_date"].isna()
    if bad_date.any():
        warnings.append(f"Removed {int(bad_date.sum())} row(s) with an invalid date.")
        df = df[~bad_date]

    blank_id = df["review_id"] == ""
    if blank_id.any():
        df.loc[blank_id, "review_id"] = [f"AUTO-{i + 1}" for i in range(int(blank_id.sum()))]
        warnings.append(f"Generated IDs for {int(blank_id.sum())} row(s) without a review_id.")

    df.loc[df["product_name"] == "", "product_name"] = "Unknown product"

    duplicate_ids = df["review_id"].duplicated()
    if duplicate_ids.any():
        warnings.append(
            f"{int(duplicate_ids.sum())} review_id value(s) are duplicated; rows were kept."
        )

    df = df.sort_values("review_date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid review rows remain after cleaning.")
    return df, warnings


def load_reviews(source: str | Path | IO | None = None) -> dict:
    """Load reviews from a CSV path or uploaded file object.

    Returns a dict with:
    - dataframe: cleaned reviews
    - source_label: human-readable description of where the data came from
    - is_sample: True when the bundled synthetic sample is being used
    - warnings: list of cleaning messages
    - error: None, or why the supplied file could not be used
    """
    result = {"dataframe": None, "source_label": "", "is_sample": False, "warnings": [], "error": None}

    if source is not None:
        name = getattr(source, "name", str(source))
        try:
            cleaned, warnings = validate_reviews(pd.read_csv(source))
            result.update(dataframe=cleaned, source_label=f"Loaded from {Path(name).name}", warnings=warnings)
            return result
        except Exception as exc:  # parser, encoding, or validation errors
            result["error"] = f"Could not use {Path(name).name}: {exc}"

    cleaned, warnings = validate_reviews(pd.read_csv(SAMPLE_CSV_PATH))
    result.update(
        dataframe=cleaned,
        source_label="Bundled synthetic sample data (not real customer reviews)",
        is_sample=True,
        warnings=warnings,
    )
    return result

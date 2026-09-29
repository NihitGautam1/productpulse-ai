"""Load and validate the reviews CSV. Build a small synthetic file if needed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "review_id",
    "product_name",
    "rating",
    "review_text",
    "review_date",
]

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CSV_PATH = DATA_DIR / "reviews.csv"


def generate_synthetic_reviews() -> pd.DataFrame:
    """Create a small labeled sample so the dashboard can run without a real CSV."""
    rows = [
        ("R001", "AeroBrew Coffee Maker", 5, "Excellent machine. Brews fast and the coffee tastes rich every morning.", "2026-01-04"),
        ("R002", "AeroBrew Coffee Maker", 2, "The carafe cracked after two weeks. Poor quality glass and weak handle.", "2026-01-09"),
        ("R003", "AeroBrew Coffee Maker", 1, "Arrived late and the box was crushed. Shipping was terrible and customer service never replied.", "2026-01-12"),
        ("R004", "PulseFit Smartwatch", 4, "Battery lasts all day and the screen is bright. Setup was easy.", "2026-01-15"),
        ("R005", "PulseFit Smartwatch", 3, "Okay watch but the heart rate reading seems average and the strap feels cheap.", "2026-01-18"),
        ("R006", "PulseFit Smartwatch", 1, "Stopped charging after a month. Defective charger and no replacement offered.", "2026-01-21"),
        ("R007", "CloudSoft Pillow", 5, "Very comfortable pillow. Sleep quality improved and the fabric is soft.", "2026-02-01"),
        ("R008", "CloudSoft Pillow", 2, "Flattened after a week. Support is poor and it smells strongly out of the package.", "2026-02-03"),
        ("R009", "CloudSoft Pillow", 4, "Good value for the price. A bit firm at first but it loosened up nicely.", "2026-02-06"),
        ("R010", "Nimbus Headphones", 5, "Noise cancellation is outstanding. Comfortable for long flights.", "2026-02-10"),
        ("R011", "Nimbus Headphones", 2, "Left ear cup rattles. Build quality is disappointing for this price.", "2026-02-14"),
        ("R012", "Nimbus Headphones", 1, "Delivery took three weeks. Package was damaged and the cable was missing.", "2026-02-16"),
        ("R013", "Nimbus Headphones", 3, "Sound is fine. Bass is average and they get warm after two hours.", "2026-02-20"),
        ("R014", "KitchenPro Blender", 5, "Blends frozen fruit smoothly. Powerful motor and easy to clean.", "2026-03-01"),
        ("R015", "KitchenPro Blender", 1, "Motor burned out on the second smoothie. Worst quality I have seen.", "2026-03-04"),
        ("R016", "KitchenPro Blender", 2, "Lid leaked all over the counter. Messy design and weak seal.", "2026-03-08"),
        ("R017", "AeroBrew Coffee Maker", 4, "Reliable daily brew. Warming plate works well. Wish it were a little quieter.", "2026-03-11"),
        ("R018", "PulseFit Smartwatch", 5, "Love the sleep tracking. Notifications are clear and the app is simple.", "2026-03-15"),
        ("R019", "CloudSoft Pillow", 1, "Customer service refused a return. Horrible experience after the zipper broke.", "2026-03-18"),
        ("R020", "KitchenPro Blender", 4, "Great for sauces. Instructions were clear and shipping was fast.", "2026-03-22"),
        ("R021", "Nimbus Headphones", 4, "Comfortable and clear calls. Case is sturdy. Price is a bit high.", "2026-03-25"),
        ("R022", "PulseFit Smartwatch", 2, "Band caused skin irritation. Cheap materials and slow software updates.", "2026-03-28"),
        ("R023", "AeroBrew Coffee Maker", 3, "Coffee is decent. Water tank is small so I refill often. Neutral overall.", "2026-04-02"),
        ("R024", "KitchenPro Blender", 5, "Replaced my old blender. Ice crushing is excellent and cleanup is quick.", "2026-04-06"),
    ]
    return pd.DataFrame(
        rows,
        columns=REQUIRED_COLUMNS,
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def validate_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep usable rows and collect warnings for missing or invalid values."""
    warnings: list[str] = []
    df = _normalize_columns(df)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing)
            + ". Expected: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    df = df[REQUIRED_COLUMNS].copy()
    before = len(df)
    df = df.dropna(how="all")
    if len(df) < before:
        warnings.append(f"Dropped {before - len(df)} completely empty row(s).")

    df["review_id"] = df["review_id"].astype(str).str.strip()
    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["review_text"] = df["review_text"].astype(str).str.strip()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    invalid_ratings = df["rating"].isna().sum()
    if invalid_ratings:
        warnings.append(f"Removed {int(invalid_ratings)} row(s) with invalid ratings.")
        df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].clip(lower=1, upper=5)

    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    invalid_dates = df["review_date"].isna().sum()
    if invalid_dates:
        warnings.append(f"Removed {int(invalid_dates)} row(s) with invalid dates.")
        df = df.dropna(subset=["review_date"])

    empty_text = (df["review_text"] == "") | (df["review_text"].str.lower() == "nan")
    if empty_text.any():
        warnings.append(f"Removed {int(empty_text.sum())} row(s) with empty review text.")
        df = df.loc[~empty_text]

    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid review rows remain after cleaning the CSV.")
    return df, warnings


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_reviews(csv_path: str | Path | None = None) -> dict:
    """
    Load reviews from CSV.

    Returns a dictionary with:
    - dataframe: cleaned reviews
    - source_label: human-readable source
    - is_synthetic: True when sample data was generated
    - warnings: cleanup messages
    - error: None, or a message if the real file could not be used
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    result = {
        "dataframe": None,
        "source_label": "",
        "is_synthetic": False,
        "warnings": [],
        "error": None,
        "path": str(path),
    }

    if not path.exists():
        synthetic = generate_synthetic_reviews()
        save_csv(synthetic, path)
        cleaned, warnings = validate_reviews(synthetic)
        result.update(
            {
                "dataframe": cleaned,
                "source_label": "Synthetic sample dataset (generated because no CSV was found)",
                "is_synthetic": True,
                "warnings": warnings
                + [
                    f"Created a sample file at {path}. Replace it with your own seller reviews when ready."
                ],
            }
        )
        return result

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # pandas parser errors, encoding issues, empty files
        synthetic = generate_synthetic_reviews()
        cleaned, warnings = validate_reviews(synthetic)
        result.update(
            {
                "dataframe": cleaned,
                "source_label": "Synthetic sample dataset (CSV could not be read)",
                "is_synthetic": True,
                "warnings": warnings,
                "error": f"Could not read {path}: {exc}",
            }
        )
        return result

    try:
        cleaned, warnings = validate_reviews(raw)
    except ValueError as exc:
        synthetic = generate_synthetic_reviews()
        cleaned, synth_warnings = validate_reviews(synthetic)
        result.update(
            {
                "dataframe": cleaned,
                "source_label": "Synthetic sample dataset (CSV was invalid)",
                "is_synthetic": True,
                "warnings": synth_warnings,
                "error": str(exc),
            }
        )
        return result

    result.update(
        {
            "dataframe": cleaned,
            "source_label": f"Loaded from {path.name}",
            "is_synthetic": False,
            "warnings": warnings,
        }
    )
    return result

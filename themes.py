"""Find common complaint themes from review text. Counts come from the data."""

from __future__ import annotations

import re

import pandas as pd

# Simple seller-focused themes. A review can match more than one theme.
COMPLAINT_THEMES = {
    "Shipping & delivery": [
        "late",
        "delayed",
        "shipping",
        "delivery",
        "package",
        "damaged box",
        "crushed",
        "missing",
    ],
    "Product quality": [
        "quality",
        "broke",
        "broken",
        "cracked",
        "defective",
        "cheap",
        "burned out",
        "rattles",
        "leaked",
        "weak",
    ],
    "Customer service": [
        "customer service",
        "support",
        "never replied",
        "refused",
        "return",
        "replacement",
    ],
    "Price & value": [
        "price",
        "expensive",
        "overpriced",
        "value",
        "cost",
    ],
    "Comfort & materials": [
        "uncomfortable",
        "irritation",
        "smells",
        "fabric",
        "strap",
        "band",
        "materials",
    ],
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def match_themes(text: str) -> list[str]:
    blob = _normalize(text)
    matched = []
    for theme, keywords in COMPLAINT_THEMES.items():
        if any(keyword in blob for keyword in keywords):
            matched.append(theme)
    return matched


def top_complaints(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """
    Rank complaint themes using negative reviews (and 1–2 star reviews).

    Returns theme, mention_count, and example_review calculated from the dataset.
    """
    if df.empty:
        return pd.DataFrame(columns=["theme", "mention_count", "example_review"])

    complaint_rows = df[
        (df["sentiment"] == "Negative") | (df["rating"] <= 2)
    ].copy()
    if complaint_rows.empty:
        return pd.DataFrame(columns=["theme", "mention_count", "example_review"])

    records: dict[str, dict] = {
        theme: {"mention_count": 0, "example_review": ""}
        for theme in COMPLAINT_THEMES
    }

    for _, row in complaint_rows.iterrows():
        text = str(row["review_text"])
        for theme in match_themes(text):
            records[theme]["mention_count"] += 1
            if not records[theme]["example_review"]:
                snippet = text if len(text) <= 140 else text[:137] + "..."
                records[theme]["example_review"] = snippet

    table = (
        pd.DataFrame(
            [
                {
                    "theme": theme,
                    "mention_count": values["mention_count"],
                    "example_review": values["example_review"],
                }
                for theme, values in records.items()
                if values["mention_count"] > 0
            ]
        )
        .sort_values(["mention_count", "theme"], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )
    return table

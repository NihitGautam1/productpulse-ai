"""Keyword-based complaint theme detection.

This is a transparent keyword matcher, not a machine-learning topic model.
Keywords are matched as whole words/phrases, so "late" does not match "plate".
"""

from __future__ import annotations

import re

import pandas as pd

# A review can match more than one theme.
COMPLAINT_THEMES: dict[str, list[str]] = {
    "Shipping & Delivery": [
        "late", "delayed", "delay", "shipping", "delivery", "delivered", "courier",
        "damaged box", "crushed", "arrived damaged", "missing",
    ],
    "Product Quality": [
        "quality", "broke", "broken", "cracked", "defective", "faulty", "cheap",
        "burned out", "rattles", "leaked", "leaks", "weak", "stopped working",
    ],
    "Customer Service": [
        "customer service", "customer support", "support team", "contacted support",
        "never replied", "no response", "did not respond", "refused", "refund",
        "return", "replacement",
    ],
    "Price & Value": ["price", "expensive", "overpriced", "value", "cost", "worth"],
    "Comfort & Materials": [
        "uncomfortable", "irritation", "smells", "fabric", "strap", "band",
        "materials", "material", "tight",
    ],
}

_PATTERNS = {
    theme: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)
    for theme, keywords in COMPLAINT_THEMES.items()
}


def match_themes(text: str) -> list[str]:
    """Return every theme whose keywords appear in the text."""
    return [theme for theme, pattern in _PATTERNS.items() if pattern.search(str(text))]


def complaint_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Reviews treated as complaints: VADER Negative or rated 1-2 stars."""
    return df[(df["sentiment"] == "Negative") | (df["rating"] <= 2)]


def top_complaints(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Rank complaint themes by how many complaint reviews mention them.

    Returns columns: theme, mentions, example_review (taken from the dataset).
    """
    columns = ["theme", "mentions", "example_review"]
    complaints = complaint_rows(df)
    if complaints.empty:
        return pd.DataFrame(columns=columns)

    records = []
    for theme, pattern in _PATTERNS.items():
        hits = complaints[complaints["review_text"].str.contains(pattern, na=False)]
        if not hits.empty:
            records.append(
                {"theme": theme, "mentions": len(hits), "example_review": hits.iloc[0]["review_text"]}
            )
    if not records:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(records, columns=columns)
        .sort_values(["mentions", "theme"], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )

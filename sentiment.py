"""Score each review with VADER. Runs locally on CPU with no API keys."""

from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer: SentimentIntensityAnalyzer | None = None


def get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def label_from_compound(compound: float) -> str:
    """VADER recommended thresholds for positive / negative / neutral."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def add_sentiment(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    """Add compound score and sentiment label for every review."""
    analyzer = get_analyzer()
    scored = df.copy()
    compounds: list[float] = []
    labels: list[str] = []

    for text in scored[text_column].astype(str):
        scores = analyzer.polarity_scores(text)
        compound = float(scores["compound"])
        compounds.append(compound)
        labels.append(label_from_compound(compound))

    scored["sentiment_score"] = compounds
    scored["sentiment"] = labels
    return scored


def sentiment_counts(df: pd.DataFrame) -> pd.Series:
    order = ["Positive", "Neutral", "Negative"]
    counts = df["sentiment"].value_counts()
    return counts.reindex(order, fill_value=0)

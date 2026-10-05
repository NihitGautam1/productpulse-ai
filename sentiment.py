"""Score each review with VADER. Runs locally on CPU with no API key."""

from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]

_analyzer: SentimentIntensityAnalyzer | None = None


def get_analyzer() -> SentimentIntensityAnalyzer:
    """Create the VADER analyzer once and reuse it."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def label_from_compound(compound: float) -> str:
    """VADER's recommended thresholds: >= 0.05 positive, <= -0.05 negative, else neutral."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def add_sentiment(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    """Return a copy of df with `sentiment_score` (VADER compound) and `sentiment` label columns."""
    analyzer = get_analyzer()
    scored = df.copy()
    scored["sentiment_score"] = [
        float(analyzer.polarity_scores(text)["compound"]) for text in scored[text_column].astype(str)
    ]
    scored["sentiment"] = scored["sentiment_score"].map(label_from_compound)
    return scored


def sentiment_counts(df: pd.DataFrame) -> pd.Series:
    """Count reviews per sentiment label, always in Positive/Neutral/Negative order."""
    return df["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)

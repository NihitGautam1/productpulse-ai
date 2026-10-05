"""Find the reviews most relevant to a question (used by Ask Your Reviews).

Uses TF-IDF text similarity (scikit-learn), boosted by theme matches, so the AI
receives the reviews that actually relate to the question instead of the whole dataset.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from themes import match_themes

QUESTION_HINTS = {
    "Shipping & Delivery": ["deliver", "shipping", "courier", "late", "arriv"],
    "Price & Value": ["price", "value", "worth", "expensive", "cost", "money"],
    "Customer Service": ["service", "support", "refund", "return", "replacement", "warranty"],
    "Battery & Charging": ["battery", "charg"],
    "Connectivity": ["bluetooth", "connect", "pair", "disconnect"],
    "Product Quality": ["quality", "durab", "broke", "defect"],
    "Comfort & Materials": ["comfort", "fit", "strap", "band"],
}


def question_themes(question: str) -> list[str]:
    """Themes the question is about, from theme keywords plus extra hints."""
    found = set(match_themes(question))
    lowered = question.lower()
    for theme, hints in QUESTION_HINTS.items():
        if any(h in lowered for h in hints):
            found.add(theme)
    return sorted(found)


def find_relevant_reviews(df: pd.DataFrame, question: str, mentions: pd.DataFrame | None = None,
                          limit: int = 60) -> pd.DataFrame:
    """Return up to `limit` reviews ranked by relevance to the question (column `relevance`).

    If the question is general ("what do customers like most?"), the result is filled with a
    balanced sample: the strongest positive and negative reviews plus the most recent ones.
    """
    if df.empty:
        return df.assign(relevance=[])
    texts = df["review_text"].astype(str)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    try:
        matrix = vectorizer.fit_transform(pd.concat([texts, pd.Series([question])], ignore_index=True))
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except ValueError:  # question made only of stop words
        scores = [0.0] * len(df)
    ranked = df.assign(relevance=scores)

    themes = question_themes(question)
    if themes and mentions is not None and not mentions.empty:
        theme_ids = set(mentions.loc[mentions["theme"].isin(themes), "review_id"])
        ranked.loc[ranked["review_id"].isin(theme_ids), "relevance"] += 0.5

    relevant = ranked[ranked["relevance"] > 0.05].sort_values("relevance", ascending=False).head(limit)
    if len(relevant) < limit:
        remaining = ranked.drop(relevant.index)
        per_group = max((limit - len(relevant)) // 3, 1)
        balanced = pd.concat([
            remaining.nsmallest(per_group, "sentiment_score"),
            remaining.nlargest(per_group, "sentiment_score"),
            remaining.sort_values("review_date", ascending=False).head(per_group),
        ]).drop_duplicates("review_id")
        relevant = pd.concat([relevant, balanced]).drop_duplicates("review_id").head(limit)
    return relevant

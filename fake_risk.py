"""Fake-review RISK indicators.

This is a risk indicator, not proof that a review is fake. Genuine customers can
trigger these signals (e.g. a short "Good product."). Use the score to decide which
reviews deserve a closer look, never to accuse a reviewer.

Signals and points (risk score is capped at 100):

    Near-duplicate text      +40  almost identical to another review (similarity >= 0.90)
    Very similar text        +20  highly similar to another review (similarity >= 0.75)
                                  (similarity only counts between reviews of similar length)
    Review burst             +20  3+ reviews of the same product, same day, same rating,
                                  well above that product's normal daily volume
    Generic text             +15  6 words or fewer and no specific product detail
    Promotional phrasing     +10  phrases like "must buy", "best ever", "highly recommended"
    Rating/text mismatch     +15  4-5 star text that reads clearly negative, or 1-2 star text that reads
                                  clearly positive (mixed "good but bad" reviews are excluded)
    Shouting                 +10  "!!!" or mostly capital letters
    Extreme rating           +5   1 or 5 stars, added only when another signal is present

Levels: High >= 55, Medium >= 30, Low < 30. A single weak signal never reaches Medium.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from themes import match_themes

DISCLAIMER = "This is a risk indicator, not proof that a review is fake."
MAX_REVIEWS_FOR_SIMILARITY = 4000
PROMO_PHRASES = re.compile(
    r"must buy|best (?:\w+ )?ever|highly recommend|100% recommend|buy it now|five stars|5 stars|"
    r"worst (?:\w+ )?ever|don'?t buy|never buy|value for money|best product",
    re.IGNORECASE,
)
CONTRAST_WORDS = re.compile(r"\b(?:but|however|although|though|lekin|magar)\b|लेकिन|मगर", re.IGNORECASE)


def similarity_signals(texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """For each review, the highest text similarity to any other review and the index of that review."""
    count = len(texts)
    best = np.zeros(count)
    partner = np.full(count, -1)
    if count < 2:
        return best, partner
    normalized = (texts.str.lower().str.replace(r"[^\w\sऀ-ॿ]", " ", regex=True)
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    # Token pattern includes Devanagari vowel signs, which \w alone does not match.
    vectors = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                              token_pattern=r"[\wऀ-ॿ]+").fit_transform(normalized)
    similarity = cosine_similarity(vectors, dense_output=False).toarray() if count <= 1500 else None
    if similarity is not None:
        np.fill_diagonal(similarity, 0)
        partner = similarity.argmax(axis=1)
        best = similarity[np.arange(count), partner]
        return best, partner
    # Larger sets: compare in blocks to limit memory.
    for start in range(0, count, 500):
        block = cosine_similarity(vectors[start:start + 500], vectors).toarray()
        for offset in range(block.shape[0]):
            block[offset, start + offset] = 0
        partner[start:start + 500] = block.argmax(axis=1)
        best[start:start + 500] = block.max(axis=1)
    return best, partner


def assess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with risk_score (0-100), risk_level, risk_reasons and similar_to columns."""
    result = df.copy()
    result["risk_score"] = 0.0
    result["risk_reasons"] = [[] for _ in range(len(result))]
    result["similar_to"] = ""
    if result.empty:
        result["risk_level"] = []
        return result

    reasons = result["risk_reasons"]

    def add(mask: pd.Series, points: float, reason) -> None:
        for index in result.index[mask]:
            result.at[index, "risk_score"] += points
            reasons[index].append(reason(index) if callable(reason) else reason)

    # Text similarity (limited to the most recent reviews for very large datasets).
    candidates = result.sort_values("review_date").tail(MAX_REVIEWS_FOR_SIMILARITY)
    best, partner = similarity_signals(candidates["review_text"].astype(str).reset_index(drop=True))
    similarity = pd.Series(best, index=candidates.index).reindex(result.index, fill_value=0.0)
    partner_ids = pd.Series(candidates["review_id"].to_numpy()[partner], index=candidates.index).reindex(result.index)
    # Only compare reviews of similar length: a short review contained in a long one is not a copy.
    lengths = result["review_text"].astype(str).str.len()
    partner_lengths = partner_ids.map(result.set_index("review_id")["review_text"].astype(str).str.len().groupby(level=0).first())
    length_ratio = (pd.concat([lengths, partner_lengths], axis=1).min(axis=1)
                    / pd.concat([lengths, partner_lengths], axis=1).max(axis=1)).fillna(0)
    similarity = similarity.where(length_ratio >= 0.6, 0.0)
    result["max_similarity"] = similarity.round(2)
    near_duplicate = similarity >= 0.90
    very_similar = (similarity >= 0.75) & ~near_duplicate
    result.loc[near_duplicate | very_similar, "similar_to"] = partner_ids[near_duplicate | very_similar]
    add(near_duplicate, 40, lambda i: f"Near-duplicate of review {partner_ids[i]} ({similarity[i]:.0%} similar)")
    add(very_similar, 20, lambda i: f"Very similar to review {partner_ids[i]} ({similarity[i]:.0%} similar)")

    # Bursts: many same-rating reviews for one product on one day, above normal volume.
    day = result["review_date"].dt.normalize()
    same_day = result.groupby([result["product_name"], day, result["rating"]])["review_id"].transform("count")
    daily = result.groupby([result["product_name"], day]).size()
    typical = daily.groupby(level=0).median()
    product_typical = result["product_name"].map(typical).fillna(1)
    burst = (same_day >= 3) & (same_day >= 3 * product_typical)
    add(burst, 20, lambda i: f"Part of a burst: {int(same_day[i])} reviews with {int(result.at[i, 'rating'])} stars for this product on {day[i]:%d %b %Y}")

    texts = result["review_text"].astype(str)
    word_counts = texts.str.split().str.len()
    no_detail = texts.map(lambda t: not match_themes(t))
    add((word_counts <= 6) & no_detail, 15, "Generic text with no specific product detail")
    add(texts.str.contains(PROMO_PHRASES), 10, "Promotional or exaggerated phrasing")

    mixed = texts.str.contains(CONTRAST_WORDS)
    mismatch = ~mixed & (((result["rating"] >= 4) & (result["sentiment_score"] <= -0.5))
                         | ((result["rating"] <= 2) & (result["sentiment_score"] >= 0.5)))
    add(mismatch, 15, "Star rating contradicts the review text")

    letters = texts.str.count(r"[A-Za-z]")
    capitals = texts.str.count(r"[A-Z]")
    shouting = texts.str.contains(r"!{3,}") | ((letters >= 10) & (capitals / letters.where(letters > 0) > 0.6))
    add(shouting, 10, "Excessive punctuation or capital letters")

    extreme = result["rating"].isin([1, 5]) & (result["risk_score"] > 0)
    add(extreme, 5, "Extreme rating combined with other signals")

    result["risk_score"] = result["risk_score"].clip(upper=100).round(0).astype(int)
    result["risk_level"] = pd.cut(result["risk_score"], bins=[-1, 29, 54, 100], labels=["Low", "Medium", "High"]).astype(str)
    return result

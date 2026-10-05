"""Theme (aspect) detection: what each review praises or complains about.

This is a transparent keyword method, not a machine-learning topic model:

1. Each review is split into clauses ("Sound is great but battery dies" -> 2 clauses).
2. Each clause is matched against theme keyword lists (whole words only).
3. The clause is a *complaint* if it contains a complaint keyword for that theme
   (e.g. "disconnecting") or its VADER score is negative; *praise* if its score is
   positive. Unclear clauses fall back to the star rating.

English, Hinglish and Hindi keywords are included.
"""

from __future__ import annotations

import re

import pandas as pd

from sentiment import score_text

# theme -> {"keywords": neutral topic words, "complaint": words that always signal a complaint}
THEMES: dict[str, dict[str, list[str]]] = {
    "Shipping & Delivery": {
        "keywords": ["delivery", "delivered", "shipping", "shipped", "courier", "package", "packaging",
                     "packed", "box", "arrived", "dispatch", "डिलीवरी"],
        "complaint": ["late", "delayed", "delay", "crushed", "damaged box", "arrived damaged", "missing",
                      "der se", "देर"],
    },
    "Product Quality": {
        "keywords": ["quality", "build", "durable", "durability", "sturdy", "hinge", "headband", "carafe",
                     "lid", "made", "material", "materials"],
        "complaint": ["broke", "broken", "cracked", "defective", "faulty", "cheap", "flimsy", "creaky",
                      "rattles", "leaked", "leaks", "leak", "stopped working", "burned out",
                      "kharab", "toot", "tut gaya", "खराब", "टूट"],
        # "sound quality" is about performance, not build quality.
        "exclude": ["sound quality", "audio quality", "call quality", "picture quality", "sleep quality",
                    "आवाज़ की गुणवत्ता"],
    },
    "Customer Service": {
        "keywords": ["customer service", "customer support", "customer care", "support team", "replacement",
                     "refund", "return", "warranty", "seller", "emails", "jawab"],
        "complaint": ["never replied", "did not respond", "didn't respond", "no response", "no reply",
                      "refused", "unresponsive", "jawab nahi", "जवाब नहीं"],
    },
    "Price & Value": {
        "keywords": ["price", "value", "cost", "worth", "rupee", "rupees", "money", "sale", "paisa", "कीमत"],
        "complaint": ["expensive", "overpriced", "not worth", "waste of money", "mehenga", "mehnga"],
    },
    "Comfort & Materials": {
        "keywords": ["comfortable", "comfort", "fit", "strap", "band", "cushions", "cushion", "ear cups",
                     "earcups", "clamping", "fabric", "soft", "pehen"],
        "complaint": ["uncomfortable", "irritation", "tight", "sweaty", "smells", "itchy", "hurts"],
    },
    "Battery & Charging": {
        "keywords": ["battery", "charge", "charging", "charger", "charges", "backup", "बैटरी"],
        "complaint": ["drains", "drain", "stopped charging", "not charging", "khatam", "खत्म"],
    },
    "Connectivity": {
        "keywords": ["bluetooth", "connection", "connect", "connectivity", "pairing", "pairs", "paired",
                     "sync", "wifi", "signal"],
        "complaint": ["disconnect", "disconnects", "disconnecting", "disconnected", "loses connection",
                      "drops", "cuts out", "cut ho", "pairing fails"],
    },
    "Performance & Features": {
        "keywords": ["sound", "audio", "bass", "noise cancellation", "anc", "vocals", "soundstage", "camera",
                     "display", "screen", "gps", "heart rate", "sleep tracking", "step counting", "app",
                     "notifications", "brews", "brew", "coffee", "taste", "motor", "performance", "features",
                     "tank", "आवाज़"],
        "complaint": ["noisy", "crashing", "crashes", "laggy", "inaccurate", "jump around", "loses track"],
    },
    "Design & Usability": {
        "keywords": ["design", "looks", "easy to use", "easy to clean", "setup", "instructions", "simple"],
        "complaint": ["confusing", "hard to use", "complicated"],
    },
}

CLAUSE_SPLIT = re.compile(r"[.!?;।]+|\b(?:but|however|although|though|lekin|magar|whereas)\b|लेकिन|मगर", re.IGNORECASE)


def _pattern(words: list[str]) -> re.Pattern:
    escaped = sorted((re.escape(w) for w in words), key=len, reverse=True)
    # \b does not work next to Devanagari vowel signs, so use look-arounds on word characters.
    return re.compile(r"(?<![\wऀ-ॿ])(?:" + "|".join(escaped) + r")(?![\wऀ-ॿ])", re.IGNORECASE)


_TOPIC = {theme: _pattern(spec["keywords"] + spec["complaint"]) for theme, spec in THEMES.items()}
_EXCLUDE = {theme: _pattern(spec["exclude"]) for theme, spec in THEMES.items() if spec.get("exclude")}
_COMPLAINT = {theme: _pattern(spec["complaint"]) for theme, spec in THEMES.items()}

MENTION_COLUMNS = ["review_id", "product_name", "review_date", "rating", "sentiment", "sentiment_score",
                   "theme", "polarity", "clause", "clause_score", "review_text"]


def split_clauses(text: str) -> list[str]:
    """Split review text into clauses at sentence ends and contrast words."""
    return [c.strip(" ,-") for c in CLAUSE_SPLIT.split(str(text)) if c and c.strip(" ,-")]


def theme_in_text(theme: str, text: str) -> bool:
    """True if the theme is mentioned, ignoring excluded phrases such as "sound quality"."""
    if theme in _EXCLUDE:
        text = _EXCLUDE[theme].sub(" ", text)
    return bool(_TOPIC[theme].search(text))


def clause_polarity(theme: str, clause: str, clause_score: float, rating: float) -> str:
    """Decide whether a theme mention is praise, a complaint, or neutral."""
    if _COMPLAINT[theme].search(clause):
        return "complaint"
    if clause_score <= -0.05:
        return "complaint"
    if clause_score >= 0.05:
        return "praise"
    if rating <= 2:
        return "complaint"
    if rating >= 4:
        return "praise"
    return "neutral"


def match_themes(text: str) -> list[str]:
    """Return every theme mentioned anywhere in the text."""
    return [theme for theme in THEMES if theme_in_text(theme, str(text))]


def extract_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (review, theme) with polarity praise/complaint/neutral and the supporting clause.

    If a review mentions a theme in several clauses, a complaint takes precedence over praise.
    """
    rows = []
    for review in df.itertuples(index=False):
        found: dict[str, dict] = {}
        for clause in split_clauses(review.review_text):
            themes_here = [theme for theme in THEMES if theme_in_text(theme, clause)]
            if not themes_here:
                continue
            clause_score = score_text(clause)
            for theme in themes_here:
                polarity = clause_polarity(theme, clause, clause_score, review.rating)
                previous = found.get(theme)
                if previous is None or (polarity == "complaint" and previous["polarity"] != "complaint"):
                    found[theme] = {"polarity": polarity, "clause": clause, "clause_score": clause_score}
        for theme, info in found.items():
            rows.append({
                "review_id": review.review_id, "product_name": review.product_name,
                "review_date": review.review_date, "rating": review.rating, "sentiment": review.sentiment,
                "sentiment_score": review.sentiment_score, "theme": theme, **info,
                "review_text": review.review_text,
            })
    return pd.DataFrame(rows, columns=MENTION_COLUMNS)


def theme_summary(mentions: pd.DataFrame, total_reviews: int) -> pd.DataFrame:
    """Per theme: reviews praising it, reviews complaining about it, and shares of all reviews."""
    columns = ["theme", "praise", "complaints", "neutral", "praise_pct", "complaint_pct"]
    if mentions.empty or not total_reviews:
        return pd.DataFrame(columns=columns)
    table = (
        mentions.pivot_table(index="theme", columns="polarity", values="review_id", aggfunc="nunique", fill_value=0)
        .reindex(columns=["praise", "complaint", "neutral"], fill_value=0)
        .rename(columns={"complaint": "complaints"})
        .reset_index()
    )
    table["praise_pct"] = (table["praise"] / total_reviews * 100).round(1)
    table["complaint_pct"] = (table["complaints"] / total_reviews * 100).round(1)
    return table[columns].sort_values(["complaints", "praise"], ascending=False).reset_index(drop=True)


def top_complaints(df: pd.DataFrame, limit: int = 5, mentions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank themes by the number of reviews complaining about them, with an example clause."""
    mentions = extract_mentions(df) if mentions is None else mentions
    complaints = mentions[mentions["polarity"] == "complaint"]
    if complaints.empty:
        return pd.DataFrame(columns=["theme", "mentions", "example_review"])
    grouped = complaints.sort_values("clause_score").groupby("theme")
    table = pd.DataFrame({
        "theme": list(grouped.groups),
        "mentions": [g["review_id"].nunique() for _, g in grouped],
        "example_review": [g.iloc[0]["review_text"] for _, g in grouped],
    })
    return table.sort_values(["mentions", "theme"], ascending=[False, True]).head(limit).reset_index(drop=True)

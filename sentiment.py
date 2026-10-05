"""Review sentiment (VADER) and language detection, with English, Hindi and Hinglish support.

VADER is an English sentiment lexicon that runs locally with no API key. To make it
more useful for product reviews and Indian-market text, we extend it with:

1. A small product-review vocabulary VADER lacks (e.g. "disconnecting", "cracked").
2. Common Hinglish (romanised Hindi) and Hindi (Devanagari) sentiment words,
   intensifiers ("bahut", "ekdum") and negations ("nahi").

This improves coverage but is NOT a full Hindi/Hinglish language model. Accuracy on
Hindi/Hinglish text is lower than on English.
"""

from __future__ import annotations

import re

import pandas as pd
from vaderSentiment import vaderSentiment as vader

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
LANGUAGE_ORDER = ["English", "Hinglish", "Hindi"]

# Product-review words missing from VADER's English lexicon (scores on VADER's -4..+4 scale).
REVIEW_LEXICON = {
    "late": -1.2, "disconnecting": -1.8, "disconnects": -1.8, "disconnect": -1.5, "disconnected": -1.5,
    "drains": -1.5, "drain": -1.2, "cracked": -1.8, "leaks": -1.5, "leaking": -1.5, "leak": -1.3,
    "overpriced": -1.8, "crashing": -1.8, "crashes": -1.8, "dented": -1.5, "creaky": -1.3,
    "cheap": -0.8, "flimsy": -1.6, "laggy": -1.5, "glitchy": -1.6, "unresponsive": -1.6,
    "respond": 0.8, "responded": 0.8, "replied": 0.8, "reliable": 1.6, "sturdy": 1.5, "crisp": 1.3, "seamless": 1.8, "durable": 1.6,
}

# Hinglish / Hindi sentiment words.
INDIC_LEXICON = {
    # positive
    "accha": 1.9, "acha": 1.9, "achha": 1.9, "acchi": 1.9, "achhi": 1.9, "achi": 1.9, "acche": 1.9,
    "badhiya": 2.2, "badiya": 2.2, "zabardast": 2.6, "mast": 2.0, "shandar": 2.5, "shaandar": 2.5,
    "behtareen": 2.6, "behtarin": 2.6, "pasand": 1.6, "maza": 1.8, "mazaa": 1.8, "sahi": 1.2,
    "अच्छा": 1.9, "अच्छी": 1.9, "अच्छे": 1.9, "बढ़िया": 2.2, "शानदार": 2.5, "बेहतरीन": 2.6, "पसंद": 1.6,
    # negative
    "bekar": -2.2, "bekaar": -2.2, "bakwas": -2.5, "bakwaas": -2.5, "ghatiya": -2.5, "faltu": -2.0,
    "kharab": -2.0, "kharaab": -2.0, "khatam": -1.5, "toot": -1.8, "tut": -1.6, "dhoka": -2.5,
    "bura": -1.9, "buri": -1.9, "pareshan": -1.8, "pareshani": -1.8,
    "खराब": -2.0, "बेकार": -2.2, "घटिया": -2.5, "खत्म": -1.5, "टूट": -1.8, "बुरा": -1.9, "परेशान": -1.8,
    "देर": -1.0,
}

INDIC_BOOSTERS = {"bahut": 0.293, "bohot": 0.293, "bahot": 0.293, "ekdum": 0.293, "kaafi": 0.2,
                  "bilkul": 0.2, "बहुत": 0.293, "एकदम": 0.293, "काफी": 0.2}
INDIC_NEGATIONS = ["nahi", "nahin", "nai", "mat", "नहीं", "न"]

# Words that indicate romanised Hindi (Hinglish).
HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nahin", "bahut", "bohot", "accha", "acha", "achha", "acchi", "achhi",
    "ekdum", "kharab", "khatam", "bekar", "bekaar", "jata", "jati", "gaya", "gayi", "thi", "tha",
    "mein", "kaafi", "badhiya", "bakwas", "ghatiya", "yaar", "bilkul", "aur", "bhi", "kya", "ho",
    "ka", "ki", "ke", "se", "ko", "jaldi", "wala", "wali", "sakte", "maza", "pehen", "diya", "koi",
}
STRONG_HINGLISH_MARKERS = HINGLISH_MARKERS - {"ho", "ka", "ki", "ke", "se", "ko", "aur", "bhi", "kya"}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Rewrites applied before scoring (text shown to users is never changed).
SENTIMENT_REWRITES = [
    # "customer care/support/service" is a topic, not praise: VADER scores "care"/"support" as positive.
    (re.compile(r"\bcustomer\s+(care|support|service)\b", re.IGNORECASE), "customerservice"),
    (re.compile(r"\bsupport\s+team\b", re.IGNORECASE), "customerservice"),
    # "koi jawab nahi diya" = "gave no reply".
    (re.compile(r"\bjawab\s+nahi\w*(\s+(diya|mila|aaya|aya)\b)?", re.IGNORECASE), "unresponsive"),
    (re.compile(r"जवाब\s+नहीं(\s+\S+)?"), "unresponsive"),
    # Hindi negation follows the word it negates ("accha nahi hai"); VADER expects it before.
    (re.compile(r"\b(\w+)\s+(nahi|nahin|nai)\b", re.IGNORECASE), r"not \1"),
    (re.compile(r"(\S+)\s+नहीं"), r"not \1"),
]

_analyzer: vader.SentimentIntensityAnalyzer | None = None


def get_analyzer() -> vader.SentimentIntensityAnalyzer:
    """Create the extended VADER analyzer once and reuse it."""
    global _analyzer
    if _analyzer is None:
        # VADER keeps boosters and negations as module-level tables; extend them once.
        vader.BOOSTER_DICT.update(INDIC_BOOSTERS)
        vader.NEGATE.extend(word for word in INDIC_NEGATIONS if word not in vader.NEGATE)
        _analyzer = vader.SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(REVIEW_LEXICON)
        _analyzer.lexicon.update(INDIC_LEXICON)
    return _analyzer


def detect_language(text: str) -> str:
    """Classify text as English, Hindi (Devanagari script) or Hinglish (romanised Hindi).

    A simple heuristic: script detection plus a list of common Hinglish words.
    """
    text = str(text)
    letters = [ch for ch in text if ch.isalpha()]
    if letters and len(DEVANAGARI.findall(text)) / len(letters) > 0.3:
        return "Hindi"
    words = re.findall(r"[a-z]+", text.lower())
    markers = [w for w in words if w in HINGLISH_MARKERS]
    strong = [w for w in markers if w in STRONG_HINGLISH_MARKERS]
    if len(strong) >= 2 or (strong and (len(markers) >= 2 or len(words) <= 5)):
        return "Hinglish"
    return "English"


def normalize_for_sentiment(text: str) -> str:
    """Apply SENTIMENT_REWRITES so VADER handles review phrasing and Hindi word order better."""
    text = str(text)
    for pattern, replacement in SENTIMENT_REWRITES:
        text = pattern.sub(replacement, text)
    return text


def score_text(text: str) -> float:
    """VADER compound score (-1 to +1) for one piece of text."""
    return float(get_analyzer().polarity_scores(normalize_for_sentiment(text))["compound"])


def label_from_compound(compound: float) -> str:
    """VADER's recommended thresholds: >= 0.05 positive, <= -0.05 negative, else neutral."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def add_sentiment(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    """Return a copy of df with `sentiment_score`, `sentiment` and `language` columns."""
    scored = df.copy()
    texts = scored[text_column].astype(str)
    scored["sentiment_score"] = [score_text(t) for t in texts]
    scored["sentiment"] = scored["sentiment_score"].map(label_from_compound)
    scored["language"] = [detect_language(t) for t in texts]
    return scored


def sentiment_counts(df: pd.DataFrame) -> pd.Series:
    """Count reviews per sentiment label, always in Positive/Neutral/Negative order."""
    return df["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)

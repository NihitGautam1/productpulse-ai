"""Emotion detection: what customers feel beyond positive or negative.

A transparent keyword approach (English, Hinglish and Hindi), like the rest of the
local analysis: every label can be traced back to the words that triggered it.
Each review gets one primary emotion, the one with the most matching words.
"""

from __future__ import annotations

import re

import pandas as pd

NO_EMOTION = "No clear emotion"
# Order matters for ties: stronger and more actionable emotions win.
EMOTIONS = {
    "Anger": {
        "emoji": "😠", "positive": False,
        "words": [r"angry", r"furious", r"worst", r"pathetic", r"terrible", r"horrible", r"scam", r"fraud",
                  r"cheated", r"ridiculous", r"unacceptable", r"never (?:again|buying|buy)", r"bakwa+s",
                  r"ghatiya", "घटिया", "बकवास"],
    },
    "Frustration": {
        "emoji": "😤", "positive": False,
        "words": [r"annoying", r"frustrat\w*", r"irritat\w*", r"keeps? \w+ing", r"again and again", r"every few",
                  r"constantly", r"fed up", r"tired of", r"hassle", r"struggl\w*", r"pareshan\w*", r"baar baar",
                  "परेशान"],
    },
    "Disappointment": {
        "emoji": "😞", "positive": False,
        "words": [r"disappoint\w*", r"let down", r"expected (?:more|better)", r"not worth", r"regret\w*",
                  r"waste of", r"underwhelm\w*", r"bekaa?r", r"u+mm?ee?d", "निराश", "बेकार"],
    },
    "Confusion": {
        "emoji": "😕", "positive": False,
        "words": [r"confus\w*", r"unclear", r"no idea", r"don'?t understand", r"can'?t figure", r"complicated",
                  r"how (?:do|to) i", r"not sure (?:how|why|what)", r"samajh nahi", "समझ नहीं"],
    },
    "Delight": {
        "emoji": "🤩", "positive": True,
        "words": [r"love[ds]?", r"amazing", r"awesome", r"fantastic", r"excellent", r"superb", r"brilliant",
                  r"delighted", r"wonderful", r"outstanding", r"blown away", r"mind[- ]?blowing", r"zabardast",
                  r"kamaal", r"mast", r"ma+za+", r"shaandaa?r", r"lajawaa?b", "लाजवाब", "शानदार", "मज़ा", "मजा"],
    },
    "Satisfaction": {
        "emoji": "🙂", "positive": True,
        "words": [r"good", r"nice", r"happy", r"satisfied", r"decent", r"worth (?:it|the)", r"value for money",
                  r"recommend\w*", r"reliable", r"solid", r"works (?:well|fine|great)", r"a+cc?h+a", r"badhiya",
                  r"theek", "अच्छा", "अच्छी", "बढ़िया", "संतुष्ट"],
    },
}
EMOTION_ORDER = list(EMOTIONS) + [NO_EMOTION]
NEGATION = re.compile(r"(?:\bnot|\bno|\bnever|n't|\bnahi|\bnahin|नहीं)\s+(?:\w+\s+)?$", re.IGNORECASE)
_PATTERNS = {
    name: re.compile(r"(?<![\w])(?:" + "|".join(spec["words"]) + r")(?![\w])", re.IGNORECASE)
    for name, spec in EMOTIONS.items()
}


def emotion_scores(text: str) -> dict[str, int]:
    """Number of matching words per emotion. Positive words right after a negation ("not good") are ignored."""
    text = str(text)
    scores = {}
    for name, pattern in _PATTERNS.items():
        count = 0
        for match in pattern.finditer(text):
            if EMOTIONS[name]["positive"] and NEGATION.search(text[max(0, match.start() - 20):match.start()]):
                continue
            count += 1
        if count:
            scores[name] = count
    return scores


def detect_emotion(text: str) -> str:
    """The review's primary emotion (most matches; ties go to the earlier emotion in EMOTIONS)."""
    scores = emotion_scores(text)
    if not scores:
        return NO_EMOTION
    return max(EMOTIONS, key=lambda name: (scores.get(name, 0), -list(EMOTIONS).index(name)))


def add_emotions(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    """Return a copy of df with an `emotion` column."""
    result = df.copy()
    result["emotion"] = [detect_emotion(t) for t in result[text_column].astype(str)]
    return result


def emotion_label(name: str) -> str:
    """Emotion with its emoji, e.g. "😤 Frustration"."""
    return f"{EMOTIONS[name]['emoji']} {name}" if name in EMOTIONS else name


def emotion_counts(df: pd.DataFrame) -> pd.Series:
    """Reviews per emotion, in the standard order (zero-filled)."""
    if "emotion" not in df.columns:
        return pd.Series(0, index=EMOTION_ORDER)
    return df["emotion"].value_counts().reindex(EMOTION_ORDER, fill_value=0)

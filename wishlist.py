"""Feature wishlist: what customers ask to be added or changed.

Finds request sentences ("I wish it had...", "please add...", "...hona chahiye") and
groups similar requests together, so the most-requested features rise to the top.
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUEST_PATTERNS = re.compile(
    r"\b(?:i\s+)?wish\b"
    r"|\bwould\s+(?:be\s+)?(?:nice|great|good|better|awesome|helpful)\s+(?:if|to)\b"
    r"|\b(?:would|i'd|i\s+would)\s+love\s+(?:an?\b|to\s+see|if)"
    r"|\bplease\s+(?:add|bring|make|include|give|allow|let)\b"
    r"|\b(?:it|they|this|you)\s+should\s+(?:have|come|include|support|add|offer)\b"
    r"|\bhope\s+(?:they|you)\s+(?:add|bring|release|make|include|introduce)\b"
    r"|\bif\s+only\b"
    r"|\b(?:needs|lacks|missing)\s+(?:an?|the|more|some)\b"
    r"|\bhona\s+chahiye\b|\bchahiye\s+tha\b|\badd\s+karo\b|\bhota\s+to\b"
    r"|चाहिए",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")
SIMILARITY_THRESHOLD = 0.45  # requests at least this similar are treated as the same feature
# Different words for the same feature, so their requests group together (applied before comparing).
SYNONYMS = {
    r"\bequali[sz]er\b|\beq\b": "equaliser",
    r"\brunning\b|\bruns\b": "run",
    r"\bdelay(?:ed)? start\b|\bschedul\w*\b": "timer",
    r"\bmulti-?point\b|\btwo devices\b|\bsame time\b": "multipoint",
    r"\bnoise cancell?(?:ing|ation)\b": "anc",
}
REQUEST_COLUMNS = ["review_id", "product_name", "review_date", "rating", "request"]


def find_requests(text: str) -> list[str]:
    """Request sentences in one review."""
    return [s.strip() for s in SENTENCE_SPLIT.split(str(text)) if s.strip() and REQUEST_PATTERNS.search(s)]


def extract_requests(df: pd.DataFrame) -> pd.DataFrame:
    """One row per request sentence, with the review it came from."""
    rows = [
        {"review_id": r.review_id, "product_name": r.product_name, "review_date": r.review_date,
         "rating": r.rating, "request": request}
        for r in df.itertuples(index=False)
        for request in find_requests(r.review_text)
    ]
    return pd.DataFrame(rows, columns=REQUEST_COLUMNS)


def group_requests(requests: pd.DataFrame) -> pd.DataFrame:
    """Group similar requests. Returns one row per group, most-requested first, with:
    request (the most common wording), reviews, products, review_ids, examples, last_date.
    """
    columns = ["request", "reviews", "products", "review_ids", "examples", "last_date"]
    if requests.empty:
        return pd.DataFrame(columns=columns)
    texts = requests["request"].str.lower()
    for pattern, replacement in SYNONYMS.items():
        texts = texts.str.replace(pattern, replacement, regex=True)
    texts = texts.str.replace(r"[^\w\s]", " ", regex=True)
    try:
        vectors = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).fit_transform(texts)
        similarity = cosine_similarity(vectors)
    except ValueError:  # every word was a stop word
        similarity = (texts.to_numpy()[:, None] == texts.to_numpy()[None, :]).astype(float)

    group = [-1] * len(requests)
    next_group = 0
    for i in range(len(requests)):  # greedy: join the first earlier group this request is similar to
        if group[i] == -1:
            group[i] = next_group
            next_group += 1
        for j in range(i + 1, len(requests)):
            if group[j] == -1 and similarity[i, j] >= SIMILARITY_THRESHOLD:
                group[j] = group[i]

    frame = requests.assign(group=group)
    rows = []
    for _, members in frame.groupby("group"):
        wording = members["request"].value_counts()
        top = wording[wording == wording.max()].index
        rows.append({
            "request": min(top, key=len),
            "reviews": members["review_id"].nunique(),
            "products": sorted(members["product_name"].unique()),
            "review_ids": members.sort_values("review_date", ascending=False)["review_id"].drop_duplicates().tolist(),
            "examples": members.drop_duplicates("request")["request"].head(3).tolist(),
            "last_date": members["review_date"].max(),
        })
    return pd.DataFrame(rows, columns=columns).sort_values(["reviews", "last_date"], ascending=False).reset_index(drop=True)

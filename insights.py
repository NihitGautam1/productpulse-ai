"""Data-driven insights: issue priority ranking, key insights and supporting evidence.

Everything here is computed directly from the review data (no AI), so every number
can be traced back to actual reviews.

Priority score (0-100), one row per complaint theme:

    score = 40 x frequency      share of all reviews that complain about the theme (20%+ = full marks)
          + 25 x negativity     how negative the complaint clauses are (average VADER score, 0..1)
          + 20 x low_ratings    share of those complaint reviews rated 1-2 stars
          + 15 x recent_rise    1 = complaint rate rose 50%+ in the recent window, 0.5 = rose, 0 = flat/fell

Levels need both a score AND enough evidence, so a single complaint is never "Critical":

    Critical: score >= 70 and >= 5 complaint reviews
    High:     score >= 50 and >= 3 complaint reviews
    Medium:   score >= 30 and >= 2 complaint reviews
    Low:      everything else (including any issue raised by only one review)
"""

from __future__ import annotations

import re

import pandas as pd

from sentiment import sentiment_counts

PRIORITY_LEVELS = ["Critical", "High", "Medium", "Low"]
WEIGHTS = {"frequency": 40, "negativity": 25, "low_ratings": 20, "recent_rise": 15}
FULL_FREQUENCY_SHARE = 0.20


def recent_window(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (recent_start, data_end). Recent = last 30 days, or the last quarter of a short date span."""
    end = df["review_date"].max()
    span_days = max((end - df["review_date"].min()).days, 1)
    days = 30 if span_days >= 120 else max(7, span_days // 4)
    return end - pd.Timedelta(days=days - 1), end


def _recent_rise(df: pd.DataFrame, theme_complaints: pd.DataFrame) -> tuple[float, float, float]:
    """Compare the theme's complaint rate in the recent window with the earlier period."""
    start, _ = recent_window(df)
    recent_reviews = (df["review_date"] >= start).sum()
    earlier_reviews = (df["review_date"] < start).sum()
    if recent_reviews < 5 or earlier_reviews < 5:
        return 0.0, 0.0, 0.0
    recent_ids = theme_complaints.loc[theme_complaints["review_date"] >= start, "review_id"].nunique()
    earlier_ids = theme_complaints.loc[theme_complaints["review_date"] < start, "review_id"].nunique()
    recent_rate, earlier_rate = recent_ids / recent_reviews, earlier_ids / earlier_reviews
    if recent_ids >= 3 and recent_rate >= 1.5 * max(earlier_rate, 0.01):
        rise = 1.0
    elif recent_ids >= 2 and recent_rate > earlier_rate:
        rise = 0.5
    else:
        rise = 0.0
    return rise, recent_rate, earlier_rate


def priority_level(score: float, complaint_reviews: int) -> str:
    """Map a score to a level, requiring enough complaints for higher levels."""
    if score >= 70 and complaint_reviews >= 5:
        return "Critical"
    if score >= 50 and complaint_reviews >= 3:
        return "High"
    if score >= 30 and complaint_reviews >= 2:
        return "Medium"
    return "Low"


def priority_ranking(df: pd.DataFrame, mentions: pd.DataFrame) -> pd.DataFrame:
    """Rank complaint themes by the transparent priority score described in the module docstring."""
    columns = ["theme", "level", "score", "complaint_reviews", "share_pct", "avg_rating", "frequency",
               "negativity", "low_ratings", "recent_rise", "recent_rate_pct", "earlier_rate_pct"]
    complaints = mentions[mentions["polarity"] == "complaint"]
    total = len(df)
    if complaints.empty or not total:
        return pd.DataFrame(columns=columns)

    rows = []
    for theme, group in complaints.groupby("theme"):
        reviews = group.drop_duplicates("review_id")
        count = len(reviews)
        frequency = min(count / total / FULL_FREQUENCY_SHARE, 1.0)
        negativity = float((-group["clause_score"]).clip(lower=0, upper=1).mean())
        low_ratings = float((reviews["rating"] <= 2).mean())
        rise, recent_rate, earlier_rate = _recent_rise(df, reviews)
        score = (WEIGHTS["frequency"] * frequency + WEIGHTS["negativity"] * negativity
                 + WEIGHTS["low_ratings"] * low_ratings + WEIGHTS["recent_rise"] * rise)
        rows.append({
            "theme": theme, "level": priority_level(score, count), "score": round(score, 1),
            "complaint_reviews": count, "share_pct": round(count / total * 100, 1),
            "avg_rating": round(float(reviews["rating"].mean()), 2), "frequency": round(frequency, 2),
            "negativity": round(negativity, 2), "low_ratings": round(low_ratings, 2), "recent_rise": rise,
            "recent_rate_pct": round(recent_rate * 100, 1), "earlier_rate_pct": round(earlier_rate * 100, 1),
        })
    table = pd.DataFrame(rows, columns=columns)
    table["level_order"] = table["level"].map(PRIORITY_LEVELS.index)
    return (table.sort_values(["level_order", "score"], ascending=[True, False])
            .drop(columns="level_order").reset_index(drop=True))


def evidence_for_theme(mentions: pd.DataFrame, theme: str, polarity: str = "complaint", limit: int = 3) -> pd.DataFrame:
    """Real reviews that support a theme insight: strongest clauses first, then most recent."""
    rows = mentions[(mentions["theme"] == theme) & (mentions["polarity"] == polarity)]
    ascending = polarity == "complaint"  # most negative complaints / most positive praise first
    rows = rows.sort_values(["clause_score", "review_date"], ascending=[ascending, False])
    return rows.drop_duplicates("review_id").head(limit)


def _ids(frame: pd.DataFrame, limit: int = 3) -> list[str]:
    return frame["review_id"].astype(str).drop_duplicates().head(limit).tolist()


def key_insights(df: pd.DataFrame, mentions: pd.DataFrame, priority: pd.DataFrame) -> list[dict]:
    """Business-oriented insights computed from the data, each with evidence review IDs.

    Returns a list of {"kind": praise|complaint|trend|overview|isolated|language, "text": str, "evidence": [ids]}.
    """
    total = len(df)
    if not total:
        return []
    insights: list[dict] = []
    counts = sentiment_counts(df)
    insights.append({
        "kind": "overview",
        "text": (f"{counts['Positive'] / total:.0%} of {total} reviews are positive and "
                 f"{counts['Negative'] / total:.0%} negative; the average rating is {df['rating'].mean():.2f}/5."),
        "evidence": [],
    })

    praise = mentions[mentions["polarity"] == "praise"]
    praise_counts = praise.groupby("theme")["review_id"].nunique().sort_values(ascending=False)
    for theme, count in praise_counts.head(2).items():
        if count >= 2:
            insights.append({
                "kind": "praise",
                "text": f"Customers frequently praise **{theme}**: {count} reviews ({count / total:.0%}) speak positively about it.",
                "evidence": _ids(evidence_for_theme(mentions, theme, "praise")),
            })

    recurring = priority[priority["complaint_reviews"] >= 2]
    for row in recurring.head(3).itertuples():
        insights.append({
            "kind": "complaint",
            "text": (f"**{row.theme}** is a recurring complaint ({row.level} priority): {row.complaint_reviews} reviews "
                     f"({row.share_pct}%) raise it, with an average rating of {row.avg_rating}/5."),
            "evidence": _ids(evidence_for_theme(mentions, row.theme)),
        })

    for row in priority[priority["recent_rise"] >= 1].itertuples():
        insights.append({
            "kind": "trend",
            "text": (f"**{row.theme}** complaints are rising: {row.recent_rate_pct}% of recent reviews mention it, "
                     f"versus {row.earlier_rate_pct}% earlier."),
            "evidence": _ids(evidence_for_theme(mentions, row.theme)),
        })

    low = df[df["rating"] <= 2]
    if len(low) >= 3:
        low_mentions = mentions[mentions["review_id"].isin(low["review_id"]) & (mentions["polarity"] == "complaint")]
        if not low_mentions.empty:
            driver = low_mentions.groupby("theme")["review_id"].nunique().sort_values(ascending=False)
            theme, count = driver.index[0], int(driver.iloc[0])
            insights.append({
                "kind": "complaint",
                "text": f"The biggest driver of low ratings is **{theme}**: it appears in {count} of {len(low)} reviews rated 1-2 stars.",
                "evidence": _ids(low_mentions[low_mentions["theme"] == theme].sort_values("clause_score")),
            })

    isolated = priority[priority["complaint_reviews"] == 1]
    if not isolated.empty:
        insights.append({
            "kind": "isolated",
            "text": "Isolated complaints (one review each, not yet a pattern): " + ", ".join(isolated["theme"]) + ".",
            "evidence": _ids(mentions[mentions["theme"].isin(isolated["theme"]) & (mentions["polarity"] == "complaint")]),
        })

    indic = df[df["language"].isin(["Hindi", "Hinglish"])]
    if len(indic):
        insights.append({
            "kind": "language",
            "text": f"{len(indic)} reviews ({len(indic) / total:.0%}) are written in Hindi or Hinglish.",
            "evidence": _ids(indic),
        })
    return insights


def headline(df: pd.DataFrame, mentions: pd.DataFrame, priority: pd.DataFrame) -> str:
    """One computed sentence for the top of the dashboard: the top praise and the top problem.

    Marks the praised theme with **bold** and the problem theme with __underscores__ (shown in red).
    """
    total = len(df)
    if not total:
        return "No reviews match the current filters."
    praise = mentions[mentions["polarity"] == "praise"].groupby("theme")["review_id"].nunique()
    praise = praise[praise >= 2].sort_values(ascending=False)
    loved = f"Customers love **{praise.index[0]}** ({int(praise.iloc[0])} reviews)" if len(praise) else ""
    recurring = priority[priority["complaint_reviews"] >= 2] if not priority.empty else priority
    if len(recurring):
        top = recurring.iloc[0]
        problem = f"__{top['theme']}__ is the top problem ({int(top['complaint_reviews'])} reviews, {top['level']} priority)"
        if top["recent_rise"] >= 1:
            problem += f" and it is rising: {top['recent_rate_pct']}% of recent reviews mention it"
        return f"{loved}, but {problem}." if loved else f"{problem[0].upper()}{problem[1:]}."
    counts = sentiment_counts(df)
    mood = f"{counts['Positive'] / total:.0%} of reviews are positive and no recurring problem stands out"
    return f"{loved}, and {mood}." if loved else f"{mood[0].upper()}{mood[1:]}."


def fix_impact(df: pd.DataFrame, mentions: pd.DataFrame, priority: pd.DataFrame) -> pd.DataFrame:
    """What the average rating could become if each issue were fixed.

    For each issue, reviews complaining about it are assumed to rate like the reviews that do
    not complain about it (never lower than they already rated). It is an optimistic estimate,
    since a review may mention other problems too.
    Columns: theme, level, affected, current_avg, projected_avg, gain.
    """
    columns = ["theme", "level", "affected", "current_avg", "projected_avg", "gain"]
    if df.empty or priority.empty:
        return pd.DataFrame(columns=columns)
    current = float(df["rating"].mean())
    rows = []
    for issue in priority.itertuples():
        ids = set(mentions.loc[(mentions["theme"] == issue.theme) & (mentions["polarity"] == "complaint"), "review_id"])
        affected = df["review_id"].isin(ids)
        if not affected.any() or affected.all():
            continue
        others = float(df.loc[~affected, "rating"].mean())
        projected = df["rating"].where(~affected, df["rating"].clip(lower=others)).mean()
        rows.append({"theme": issue.theme, "level": issue.level, "affected": int(affected.sum()),
                     "current_avg": round(current, 2), "projected_avg": round(float(projected), 2),
                     "gain": round(float(projected) - current, 2)})
    return pd.DataFrame(rows, columns=columns).sort_values("gain", ascending=False).reset_index(drop=True)


# Words that carry no opinion, left out of the word clouds (English + common Hinglish).
CLOUD_STOP_WORDS = {
    "product", "really", "just", "also", "even", "still", "much", "very", "quite", "got", "get", "one", "use",
    "used", "using", "bit", "lot", "thing", "things", "day", "days", "time", "times", "would", "could", "it's",
    "i'm", "i've", "don't", "doesn't", "didn't", "can't", "won't", "isn't", "wasn't", "hai", "hain", "ka", "ki",
    "ke", "ko", "se", "aur", "bhi", "ye", "yeh", "wo", "woh", "kya", "tha", "thi", "ho", "hota", "raha", "rahi",
    "bahut", "bohot", "nahi", "nahin", "ekdum", "kaafi", "par", "mein", "main", "toh", "to", "na", "hi",
}


def top_words(mentions: pd.DataFrame, polarity: str, limit: int = 28) -> list[tuple[str, int]]:
    """Most frequent meaningful words in praise or complaint clauses, as (word, reviews using it)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    clauses = mentions[mentions["polarity"] == polarity].drop_duplicates(["review_id", "clause"])
    if clauses.empty:
        return []
    stop = ENGLISH_STOP_WORDS | CLOUD_STOP_WORDS
    counts: dict[str, set] = {}
    for review_id, clause in zip(clauses["review_id"], clauses["clause"]):
        for word in re.findall(r"[a-zऀ-ॿ][a-z'ऀ-ॿ]{2,}", str(clause).lower()):
            if word not in stop:
                counts.setdefault(word, set()).add(review_id)
    ranked = sorted(((w, len(ids)) for w, ids in counts.items()), key=lambda item: (-item[1], item[0]))
    return [(w, n) for w, n in ranked if n >= 2][:limit]


def health_score(df: pd.DataFrame) -> int:
    """0-100 summary of customer satisfaction: half average rating, half net sentiment."""
    if df.empty:
        return 0
    counts = sentiment_counts(df)
    total = len(df)
    rating_part = (df["rating"].mean() - 1) / 4
    sentiment_part = ((counts["Positive"] - counts["Negative"]) / total + 1) / 2
    return int(round(100 * (0.5 * rating_part + 0.5 * sentiment_part)))


def build_stats_context(df: pd.DataFrame, mentions: pd.DataFrame, priority: pd.DataFrame,
                        alerts: list[dict] | None = None, monthly_themes: pd.DataFrame | None = None) -> str:
    """Plain-text statistics given to the AI so its answers stay consistent with the computed numbers."""
    total = len(df)
    counts = sentiment_counts(df)
    lines = [
        f"Reviews: {total} (from {df['review_date'].min():%Y-%m-%d} to {df['review_date'].max():%Y-%m-%d})",
        f"Average rating: {df['rating'].mean():.2f}/5",
        "Rating counts: " + ", ".join(f"{int(k)} stars: {v}" for k, v in df["rating"].value_counts().sort_index().items()),
        f"Sentiment (VADER): Positive {counts['Positive']}, Neutral {counts['Neutral']}, Negative {counts['Negative']}",
        "Languages: " + ", ".join(f"{k}: {v}" for k, v in df["language"].value_counts().items()),
        "",
        "Theme mentions (number of reviews praising / complaining):",
    ]
    for theme, group in mentions.groupby("theme"):
        praise = group.loc[group["polarity"] == "praise", "review_id"].nunique()
        complaint = group.loc[group["polarity"] == "complaint", "review_id"].nunique()
        lines.append(f"- {theme}: {praise} praise, {complaint} complaints")
    if not priority.empty:
        lines += ["", "Issue priority (computed score 0-100):"]
        for row in priority.itertuples():
            lines.append(f"- {row.theme}: {row.level} (score {row.score}, {row.complaint_reviews} complaint reviews, "
                         f"recent complaint rate {row.recent_rate_pct}% vs earlier {row.earlier_rate_pct}%)")
    if monthly_themes is not None and not monthly_themes.empty:
        lines += ["", "Complaint reviews per theme per period:"]
        lines.append(monthly_themes.to_string())
    if alerts:
        lines += ["", "Detected spike alerts:"] + [f"- {a['title']}: {a['detail']}" for a in alerts]
    return "\n".join(lines)

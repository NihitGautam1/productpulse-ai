"""Time-based analysis: review trends and spike alerts.

Spike alerts compare a RECENT window with a BASELINE window before it:

    recent   = last 14 days of data (or the last quarter of the date span if it is short)
    baseline = up to 8 weeks before the recent window

Minimum data rules (no alerts are raised below these, to avoid false alarms):
    - at least 30 reviews in total
    - at least 8 reviews in the recent window and 15 in the baseline
    - each alert also needs a minimum count of affected reviews (see thresholds below)

Alerts are detected patterns, not proof of a cause.
"""

from __future__ import annotations

import pandas as pd

FREQUENCIES = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}
MIN_TOTAL_REVIEWS = 30
MIN_RECENT_REVIEWS = 8
MIN_BASELINE_REVIEWS = 15


def auto_frequency(df: pd.DataFrame) -> str:
    """Pick a readable grouping for the date span: daily, weekly or monthly."""
    span = (df["review_date"].max() - df["review_date"].min()).days if len(df) else 0
    if span <= 45:
        return "Daily"
    if span <= 240:
        return "Weekly"
    return "Monthly"


def _period(dates: pd.Series, frequency: str) -> pd.Series:
    code = FREQUENCIES[frequency]
    if code == "D":
        return dates.dt.normalize()
    if code == "MS":
        return dates.dt.to_period("M").dt.start_time
    return dates.dt.to_period("W-SUN").dt.start_time  # weeks starting Monday


def review_trends(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Per period: review count, positive/negative counts, negative share and average rating."""
    if df.empty:
        return pd.DataFrame(columns=["period", "reviews", "positive", "negative", "negative_pct", "avg_rating"])
    frame = df.assign(period=_period(df["review_date"], frequency))
    grouped = frame.groupby("period")
    table = pd.DataFrame({
        "reviews": grouped.size(),
        "positive": grouped["sentiment"].apply(lambda s: int((s == "Positive").sum())),
        "negative": grouped["sentiment"].apply(lambda s: int((s == "Negative").sum())),
        "low_ratings": grouped["rating"].apply(lambda s: int((s <= 2).sum())),
        "avg_rating": grouped["rating"].mean().round(2),
    })
    # Include empty periods so gaps are visible rather than hidden.
    full_range = pd.date_range(table.index.min(), table.index.max(), freq=FREQUENCIES[frequency])
    table = table.reindex(table.index.union(full_range)).fillna({"reviews": 0, "positive": 0, "negative": 0, "low_ratings": 0})
    table["negative_pct"] = (table["negative"] / table["reviews"].where(table["reviews"] > 0) * 100).round(1)
    return table.rename_axis("period").reset_index()


def theme_trends(mentions: pd.DataFrame, frequency: str, polarity: str = "complaint") -> pd.DataFrame:
    """Pivot table: periods x themes, counting reviews with that polarity."""
    rows = mentions[mentions["polarity"] == polarity]
    if rows.empty:
        return pd.DataFrame()
    rows = rows.assign(period=_period(rows["review_date"], frequency))
    table = rows.pivot_table(index="period", columns="theme", values="review_id", aggfunc="nunique", fill_value=0)
    return table.reindex(columns=table.sum().sort_values(ascending=False).index)


def alert_windows(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (baseline_start, recent_start, data_end)."""
    end = df["review_date"].max().normalize()
    span = max((end - df["review_date"].min().normalize()).days + 1, 1)
    recent_days = 14 if span >= 70 else max(7, span // 4)
    recent_start = end - pd.Timedelta(days=recent_days - 1)
    baseline_start = max(df["review_date"].min().normalize(), recent_start - pd.Timedelta(weeks=8))
    return baseline_start, recent_start, end


def _rate_alert(name: str, recent_hits: int, recent_total: int, base_hits: int, base_total: int,
                min_hits: int, min_jump: float, evidence: list[str], subject: str) -> dict | None:
    """Raise an alert when a rate (hits / reviews) jumps clearly above its baseline."""
    recent_rate, base_rate = recent_hits / recent_total, base_hits / base_total
    if recent_hits < min_hits or recent_rate - base_rate < min_jump or recent_rate < 1.5 * base_rate:
        return None
    ratio = recent_rate / base_rate if base_rate else float("inf")
    severity = "High" if (recent_rate - base_rate >= 0.25 or ratio >= 3) else "Medium"
    return {
        "severity": severity,
        "title": f"{name} increased sharply{subject}",
        "detail": (f"{recent_rate:.0%} of recent reviews ({recent_hits} of {recent_total}) vs "
                   f"{base_rate:.0%} in the baseline period ({base_hits} of {base_total})."),
        "recent_pct": round(recent_rate * 100, 1),
        "baseline_pct": round(base_rate * 100, 1),
        "evidence": evidence,
    }


def detect_spikes(df: pd.DataFrame, mentions: pd.DataFrame, subject: str = "") -> dict:
    """Detect unusual recent increases in negative reviews, low ratings, volume and theme complaints.

    Returns {"status": "ok" | "insufficient", "message": str, "alerts": [...], "window": (...)}.
    """
    suffix = f" for {subject}" if subject else ""
    if len(df) < MIN_TOTAL_REVIEWS:
        return {"status": "insufficient", "alerts": [], "window": None,
                "message": f"Spike detection needs at least {MIN_TOTAL_REVIEWS} reviews (this selection has {len(df)})."}

    baseline_start, recent_start, end = alert_windows(df)
    recent = df[df["review_date"] >= recent_start]
    baseline = df[(df["review_date"] >= baseline_start) & (df["review_date"] < recent_start)]
    if len(recent) < MIN_RECENT_REVIEWS or len(baseline) < MIN_BASELINE_REVIEWS:
        return {"status": "insufficient", "alerts": [], "window": (baseline_start, recent_start, end),
                "message": (f"Not enough reviews to compare periods reliably (recent: {len(recent)}, "
                            f"baseline: {len(baseline)}; need {MIN_RECENT_REVIEWS} and {MIN_BASELINE_REVIEWS}).")}

    def ids(frame: pd.DataFrame) -> list[str]:
        return frame.sort_values("review_date", ascending=False)["review_id"].astype(str).head(3).tolist()

    alerts = []
    recent_neg, base_neg = recent[recent["sentiment"] == "Negative"], baseline[baseline["sentiment"] == "Negative"]
    alerts.append(_rate_alert("Negative reviews", len(recent_neg), len(recent), len(base_neg), len(baseline),
                              4, 0.15, ids(recent_neg), suffix))
    recent_low, base_low = recent[recent["rating"] <= 2], baseline[baseline["rating"] <= 2]
    alerts.append(_rate_alert("Low ratings (1-2 stars)", len(recent_low), len(recent), len(base_low), len(baseline),
                              4, 0.15, ids(recent_low), suffix))

    complaints = mentions[mentions["polarity"] == "complaint"]
    for theme, group in complaints.groupby("theme"):
        recent_ids = group[group["review_date"] >= recent_start].drop_duplicates("review_id")
        base_ids = group[(group["review_date"] >= baseline_start) & (group["review_date"] < recent_start)].drop_duplicates("review_id")
        alerts.append(_rate_alert(f"{theme} complaints", len(recent_ids), len(recent), len(base_ids), len(baseline),
                                  3, 0.10, ids(recent_ids), suffix))

    recent_days = (end - recent_start).days + 1
    base_days = max((recent_start - baseline_start).days, 1)
    recent_per_day, base_per_day = len(recent) / recent_days, len(baseline) / base_days
    if len(recent) >= 10 and recent_per_day >= 2 * base_per_day:
        alerts.append({
            "severity": "Info", "title": f"Review volume spike{suffix}",
            "detail": f"{recent_per_day:.1f} reviews/day recently vs {base_per_day:.1f}/day in the baseline period.",
            "recent_pct": None, "baseline_pct": None, "evidence": [],
        })

    rating_drop = baseline["rating"].mean() - recent["rating"].mean()
    if rating_drop >= 0.5:
        alerts.append({
            "severity": "High" if rating_drop >= 1 else "Medium", "title": f"Average rating dropped{suffix}",
            "detail": f"{recent['rating'].mean():.2f}/5 recently vs {baseline['rating'].mean():.2f}/5 in the baseline period.",
            "recent_pct": None, "baseline_pct": None, "evidence": ids(recent_low),
        })

    order = {"High": 0, "Medium": 1, "Info": 2}
    alerts = sorted((a for a in alerts if a), key=lambda a: order[a["severity"]])
    return {"status": "ok", "alerts": alerts, "window": (baseline_start, recent_start, end),
            "message": (f"Compared {recent_start:%d %b %Y} – {end:%d %b %Y} ({len(recent)} reviews) with "
                        f"{baseline_start:%d %b %Y} – {recent_start - pd.Timedelta(days=1):%d %b %Y} ({len(baseline)} reviews).")}

"""Head-to-head comparison of two products: headline metrics and per-theme satisfaction."""

from __future__ import annotations

import pandas as pd

from insights import health_score
from sentiment import sentiment_counts

# Metric name -> (column in product_metrics, True if higher is better)
METRICS = {
    "Reviews": ("reviews", None),
    "Average rating": ("avg_rating", True),
    "Positive": ("positive_pct", True),
    "Negative": ("negative_pct", False),
    "Health score": ("health", True),
}
MIN_THEME_MENTIONS = 2  # a theme needs this many praise/complaint reviews to get a satisfaction score


def product_metrics(reviews: pd.DataFrame) -> dict:
    """Headline numbers for one product's reviews."""
    total = len(reviews)
    if not total:
        return {"reviews": 0, "avg_rating": 0.0, "positive_pct": 0.0, "negative_pct": 0.0, "health": 0}
    counts = sentiment_counts(reviews)
    return {
        "reviews": total,
        "avg_rating": round(float(reviews["rating"].mean()), 2),
        "positive_pct": round(counts["Positive"] / total * 100, 1),
        "negative_pct": round(counts["Negative"] / total * 100, 1),
        "health": health_score(reviews),
    }


def winner(metric: str, a: dict, b: dict) -> int:
    """0 if product A wins this metric, 1 if B wins, -1 for a tie or a metric with no winner."""
    column, higher_better = METRICS[metric]
    if higher_better is None or a[column] == b[column]:
        return -1
    a_better = a[column] > b[column]
    return 0 if a_better == higher_better else 1


def theme_table(mentions_a: pd.DataFrame, mentions_b: pd.DataFrame) -> pd.DataFrame:
    """Per theme and product: praise and complaint review counts and a 0-100 satisfaction score.

    Satisfaction = praise / (praise + complaints) × 100, only when there are enough mentions.
    """
    rows = []
    for label, mentions in (("a", mentions_a), ("b", mentions_b)):
        for theme, group in mentions.groupby("theme"):
            praise = group.loc[group["polarity"] == "praise", "review_id"].nunique()
            complaints = group.loc[group["polarity"] == "complaint", "review_id"].nunique()
            rows.append({"theme": theme, "product": label, "praise": praise, "complaints": complaints})
    if not rows:
        return pd.DataFrame(columns=["theme", "praise_a", "complaints_a", "praise_b", "complaints_b",
                                     "satisfaction_a", "satisfaction_b"])
    table = pd.DataFrame(rows).pivot_table(index="theme", columns="product", values=["praise", "complaints"],
                                           fill_value=0)
    table.columns = [f"{value}_{product}" for value, product in table.columns]
    table = table.reindex(columns=["praise_a", "complaints_a", "praise_b", "complaints_b"], fill_value=0).astype(int)
    for side in ("a", "b"):
        rated = table[f"praise_{side}"] + table[f"complaints_{side}"]
        table[f"satisfaction_{side}"] = (table[f"praise_{side}"] / rated.where(rated >= MIN_THEME_MENTIONS) * 100).round(0)
    table["total"] = table[["praise_a", "complaints_a", "praise_b", "complaints_b"]].sum(axis=1)
    return table.sort_values("total", ascending=False).drop(columns="total").reset_index()


def strengths_and_weaknesses(table: pd.DataFrame, side: str, limit: int = 3) -> tuple[list[str], list[str]]:
    """Themes this product is clearly better or worse at than the other one (by satisfaction score)."""
    other = "b" if side == "a" else "a"
    both = table.dropna(subset=[f"satisfaction_{side}", f"satisfaction_{other}"])
    gap = (both[f"satisfaction_{side}"] - both[f"satisfaction_{other}"])
    better = both.loc[gap >= 15, "theme"].tolist()
    worse = both.loc[gap <= -15, "theme"].tolist()
    order = gap.abs().sort_values(ascending=False).index
    better = [t for t in both.loc[order, "theme"] if t in better][:limit]
    worse = [t for t in both.loc[order, "theme"] if t in worse][:limit]
    return better, worse


def stats_text(name_a: str, a: dict, name_b: str, b: dict, table: pd.DataFrame) -> str:
    """Computed comparison statistics, in a compact form for the AI prompt."""
    lines = []
    for name, m in ((name_a, a), (name_b, b)):
        lines.append(f"{name}: {m['reviews']} reviews, average rating {m['avg_rating']}/5, "
                     f"{m['positive_pct']}% positive, {m['negative_pct']}% negative, health score {m['health']}/100")
    lines.append("Per theme (praise reviews / complaint reviews / satisfaction 0-100):")
    for row in table.itertuples():
        def part(side: str) -> str:
            score = getattr(row, f"satisfaction_{side}")
            return (f"{getattr(row, f'praise_{side}')}/{getattr(row, f'complaints_{side}')}/"
                    f"{'n/a' if pd.isna(score) else int(score)}")
        lines.append(f"- {row.theme}: {name_a} {part('a')} · {name_b} {part('b')}")
    return "\n".join(lines)

"""Themes, priority ranking, insights, trends, spike alerts, fake-review risk and retrieval."""

import pandas as pd

from fake_risk import assess_reviews
from insights import evidence_for_theme, health_score, key_insights, priority_level, priority_ranking
from retrieval import find_relevant_reviews
from sentiment import add_sentiment
from themes import extract_mentions, match_themes, top_complaints
from trends import detect_spikes, review_trends, theme_trends


def one_review(text: str, rating: int = 3) -> pd.DataFrame:
    return add_sentiment(pd.DataFrame([{
        "review_id": "X1", "product_name": "P", "rating": rating, "review_text": text,
        "review_date": pd.Timestamp("2026-01-01"),
    }]))


# ---------- themes ----------

def test_theme_matching_uses_whole_words():
    assert "Shipping & Delivery" in match_themes("Delivery was late")
    assert "Shipping & Delivery" not in match_themes("The warming plate works well")
    assert "Comfort & Materials" not in match_themes("My husband likes it")
    assert "Customer Service" not in match_themes("Pillow support is poor")


def test_sound_quality_is_not_product_quality():
    themes = match_themes("Sound quality is excellent with deep bass.")
    assert "Performance & Features" in themes
    assert "Product Quality" not in themes


def test_mixed_review_splits_praise_and_complaint():
    mentions = extract_mentions(one_review("Great value for the price. However, the left earcup randomly loses connection."))
    polarity = dict(zip(mentions["theme"], mentions["polarity"]))
    assert polarity["Price & Value"] == "praise"
    assert polarity["Connectivity"] == "complaint"


def test_mixed_hinglish_review():
    mentions = extract_mentions(one_review("Sound quality ekdum mast hai. But battery bahut jaldi khatam ho jati hai."))
    polarity = dict(zip(mentions["theme"], mentions["polarity"]))
    assert polarity["Performance & Features"] == "praise"
    assert polarity["Battery & Charging"] == "complaint"


def test_top_complaints_counts_only_complaints():
    df = add_sentiment(pd.DataFrame({
        "review_id": ["A", "B", "C"], "product_name": ["P"] * 3,
        "review_text": ["Delivery was late and box crushed", "Fast delivery, love it", "Cracked, poor quality"],
        "rating": [1, 5, 2], "review_date": pd.to_datetime(["2026-01-01"] * 3),
    }))
    counts = dict(zip(*top_complaints(df)[["theme", "mentions"]].T.values))
    assert counts.get("Shipping & Delivery") == 1
    assert counts.get("Product Quality") == 1


# ---------- priority ----------

def test_single_complaint_is_never_above_low():
    assert priority_level(100, 1) == "Low"
    assert priority_level(100, 2) == "Medium"
    assert priority_level(100, 4) == "High"
    assert priority_level(100, 5) == "Critical"


def test_priority_ranking_on_demo(demo_reviews, demo_mentions):
    priority = priority_ranking(demo_reviews, demo_mentions)
    assert not priority.empty
    assert priority["score"].between(0, 100).all()
    assert priority.iloc[0]["level"] in {"Critical", "High"}
    # Connectivity complaints rise in September in the demo data.
    connectivity = priority.set_index("theme").loc["Connectivity"]
    assert connectivity["recent_rise"] == 1.0
    assert (priority.loc[priority["complaint_reviews"] == 1, "level"] == "Low").all()


def test_priority_with_one_complaint_review():
    df = one_review("The headband cracked near the hinge.", rating=1)
    priority = priority_ranking(df, extract_mentions(df))
    assert (priority["level"] == "Low").all()


# ---------- insights & evidence ----------

def test_key_insights_evidence_exists(demo_reviews, demo_mentions):
    priority = priority_ranking(demo_reviews, demo_mentions)
    insights = key_insights(demo_reviews, demo_mentions, priority)
    assert insights and insights[0]["kind"] == "overview"
    valid_ids = set(demo_reviews["review_id"])
    for insight in insights:
        assert set(insight["evidence"]) <= valid_ids
    assert any(i["kind"] == "complaint" and i["evidence"] for i in insights)


def test_evidence_comes_from_dataset(demo_mentions, demo_reviews):
    evidence = evidence_for_theme(demo_mentions, "Connectivity", limit=3)
    assert len(evidence) == 3
    texts = dict(zip(demo_reviews["review_id"], demo_reviews["review_text"]))
    for row in evidence.itertuples():
        assert row.clause in texts[row.review_id]


def test_health_score_range(sample_reviews, demo_reviews):
    assert 0 <= health_score(sample_reviews) <= 100
    assert 0 <= health_score(demo_reviews) <= 100


# ---------- trends ----------

def test_review_trends_shape(demo_reviews):
    weekly = review_trends(demo_reviews, "Weekly")
    assert {"period", "reviews", "positive", "negative", "negative_pct", "avg_rating"} <= set(weekly.columns)
    assert weekly["reviews"].sum() == len(demo_reviews)
    monthly = review_trends(demo_reviews, "Monthly")
    assert len(monthly) == 6


def test_theme_trends_shape(demo_mentions):
    table = theme_trends(demo_mentions, "Monthly")
    assert "Connectivity" in table.columns
    assert table.loc[table.index.max(), "Connectivity"] == table["Connectivity"].max()


# ---------- spike alerts ----------

def test_spikes_need_enough_data(sample_reviews):
    result = detect_spikes(sample_reviews, extract_mentions(sample_reviews))
    assert result["status"] == "insufficient"
    assert result["alerts"] == []


def test_spike_detected_on_demo(demo_reviews, demo_mentions):
    result = detect_spikes(demo_reviews, demo_mentions)
    assert result["status"] == "ok"
    titles = [a["title"] for a in result["alerts"]]
    assert any("Connectivity complaints" in t for t in titles)
    valid_ids = set(demo_reviews["review_id"])
    for alert in result["alerts"]:
        assert set(alert["evidence"]) <= valid_ids


# ---------- fake review risk ----------

def test_burst_flagged_high_and_most_reviews_low(demo_reviews):
    risk = assess_reviews(demo_reviews)
    burst = risk[(risk["product_name"] == "PulseFit Smartwatch")
                 & (risk["review_date"] == pd.Timestamp("2026-08-14"))
                 & risk["review_text"].str.contains("ever", case=False)]
    assert len(burst) >= 5
    assert (burst["risk_level"] == "High").sum() >= 4
    assert (risk["risk_level"] == "Low").mean() > 0.75
    assert all(isinstance(r, list) for r in risk["risk_reasons"])


def test_single_genuine_review_is_low_risk():
    risk = assess_reviews(one_review("The battery lasts about two days and the strap is comfortable.", rating=4))
    assert risk.iloc[0]["risk_level"] == "Low"


# ---------- retrieval ----------

def test_retrieval_finds_delivery_reviews(demo_reviews, demo_mentions):
    results = find_relevant_reviews(demo_reviews, "Are delivery complaints increasing?", demo_mentions, limit=10)
    assert len(results) == 10
    delivery_hits = results["review_text"].str.contains("deliver|shipping|package", case=False).sum()
    assert delivery_hits >= 8


def test_retrieval_general_question_returns_balanced_sample(demo_reviews, demo_mentions):
    results = find_relevant_reviews(demo_reviews, "What do customers like most?", demo_mentions, limit=30)
    assert len(results) > 0
    assert set(results["sentiment"]) >= {"Positive", "Negative"}

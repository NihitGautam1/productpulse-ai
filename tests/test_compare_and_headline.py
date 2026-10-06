"""Product comparison, the dashboard headline and the AI headline/verdict prompts (no real API calls)."""

import pandas as pd
import pytest

import ai_service
import compare
from insights import headline, priority_ranking


@pytest.fixture(scope="module")
def two_products(demo_reviews, demo_mentions):
    names = demo_reviews["product_name"].value_counts().index[:2].tolist()
    parts = []
    for name in names:
        reviews = demo_reviews[demo_reviews["product_name"] == name]
        parts.append((name, reviews, demo_mentions[demo_mentions["review_id"].isin(reviews["review_id"])]))
    return parts


def test_product_metrics_and_winner(two_products):
    (name_a, reviews_a, _), (name_b, reviews_b, _) = two_products
    a, b = compare.product_metrics(reviews_a), compare.product_metrics(reviews_b)
    assert a["reviews"] == len(reviews_a) and 1 <= a["avg_rating"] <= 5 and 0 <= a["health"] <= 100
    assert compare.winner("Reviews", a, b) == -1  # volume has no winner
    expected = 0 if a["avg_rating"] > b["avg_rating"] else 1 if b["avg_rating"] > a["avg_rating"] else -1
    assert compare.winner("Average rating", a, b) == expected
    # Lower negative share wins.
    assert compare.winner("Negative", {"negative_pct": 5.0}, {"negative_pct": 9.0}) == 0


def test_theme_table_satisfaction(two_products):
    (_, _, mentions_a), (_, _, mentions_b) = two_products
    table = compare.theme_table(mentions_a, mentions_b)
    assert not table.empty
    scored = table.dropna(subset=["satisfaction_a"])
    for row in scored.itertuples():
        assert row.praise_a + row.complaints_a >= compare.MIN_THEME_MENTIONS
        assert row.satisfaction_a == round(row.praise_a / (row.praise_a + row.complaints_a) * 100)


def test_strengths_are_real_gaps():
    table = pd.DataFrame({"theme": ["Battery", "Comfort", "Price"], "satisfaction_a": [90, 40, 50],
                          "satisfaction_b": [50, 80, 55]})
    better_a, worse_a = compare.strengths_and_weaknesses(table, "a")
    assert better_a == ["Battery"] and worse_a == ["Comfort"]
    assert compare.strengths_and_weaknesses(table, "b")[0] == ["Comfort"]


def test_stats_text_mentions_both_products(two_products):
    (name_a, reviews_a, mentions_a), (name_b, reviews_b, mentions_b) = two_products
    table = compare.theme_table(mentions_a, mentions_b)
    text = compare.stats_text(name_a, compare.product_metrics(reviews_a), name_b, compare.product_metrics(reviews_b), table)
    assert name_a in text and name_b in text and "satisfaction" in text


def test_headline_names_praise_and_top_problem(demo_reviews, demo_mentions):
    priority = priority_ranking(demo_reviews, demo_mentions)
    text = headline(demo_reviews, demo_mentions, priority)
    assert text.startswith("Customers love **") and priority.iloc[0]["theme"] in text
    assert headline(demo_reviews.iloc[0:0], demo_mentions.iloc[0:0], priority.iloc[0:0]).startswith("No reviews")


def test_ai_headline_strips_citations(sample_reviews, monkeypatch):
    monkeypatch.setattr(ai_service, "_call_model",
                        lambda system, prompt, max_tokens=2000, media=None: "Customers love delivery. [R003, R012]")
    result = ai_service.generate_headline(sample_reviews, "all products")
    assert result["text"] == "Customers love delivery." and result["cited_ids"] == ["R003", "R012"]


def test_ai_compare_sends_both_products(demo_reviews, monkeypatch, two_products):
    (name_a, reviews_a, _), (name_b, reviews_b, _) = two_products
    prompts = []

    def fake(system, prompt, max_tokens=2000, media=None):
        prompts.append(prompt)
        return "### Verdict\nFine."

    monkeypatch.setattr(ai_service, "_call_model", fake)
    ai_service.compare_products(reviews_a, name_a, reviews_b, name_b, "stats here")
    assert f"Where {name_a} wins" in prompts[-1] and f"Where {name_b} wins" in prompts[-1]
    assert "stats here" in prompts[-1]
    with pytest.raises(ai_service.AIServiceError):
        ai_service.compare_products(reviews_a.iloc[0:0], name_a, reviews_b, name_b, "")

"""Tests for data loading, sentiment, themes and the AI service (no real API calls)."""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai_service  # noqa: E402
from data_loader import load_reviews, validate_reviews  # noqa: E402
from sentiment import add_sentiment, label_from_compound  # noqa: E402
from themes import match_themes, top_complaints  # noqa: E402


def make_csv(text: str, name: str = "test.csv") -> io.BytesIO:
    buffer = io.BytesIO(text.encode("utf-8"))
    buffer.name = name
    return buffer


# ---------- data_loader ----------

def test_sample_data_loads_and_is_labelled():
    loaded = load_reviews()
    assert loaded["is_sample"]
    assert len(loaded["dataframe"]) == 24
    assert loaded["error"] is None


def test_date_alias_and_cleaning():
    csv = (
        "review_id,product_name,rating,review_text,date\n"
        "A1,Phone,5,Great camera,2026-01-01\n"
        "A2,Phone,9,Out of range rating,2026-01-02\n"
        "A3,Phone,abc,Bad rating,2026-01-03\n"
        "A4,Phone,4,,2026-01-04\n"
        "A5,Phone,2,Bad date,not-a-date\n"
    )
    df, warnings = validate_reviews(pd.read_csv(make_csv(csv)))
    assert df["review_id"].tolist() == ["A1"]
    assert len(warnings) == 3


def test_invalid_upload_falls_back_to_sample_without_writing_files(tmp_path):
    loaded = load_reviews(make_csv("foo,bar\n1,2\n", "bad.csv"))
    assert loaded["is_sample"]
    assert "missing required column" in loaded["error"]
    # A missing path must not create a file.
    missing = tmp_path / "nope.csv"
    load_reviews(missing)
    assert not missing.exists()


# ---------- sentiment ----------

def test_sentiment_labels():
    assert label_from_compound(0.5) == "Positive"
    assert label_from_compound(-0.5) == "Negative"
    assert label_from_compound(0.0) == "Neutral"
    df = add_sentiment(pd.DataFrame({"review_text": ["I love it, excellent!", "Terrible, awful product."]}))
    assert df["sentiment"].tolist() == ["Positive", "Negative"]


# ---------- themes ----------

def test_theme_matching_uses_whole_words():
    assert "Shipping & Delivery" in match_themes("Delivery was late")
    assert "Shipping & Delivery" not in match_themes("The warming plate works well")
    assert "Comfort & Materials" not in match_themes("My husband likes it")
    assert "Customer Service" not in match_themes("Pillow support is poor")


def test_top_complaints_counts_only_complaints():
    df = pd.DataFrame(
        {
            "review_text": ["Delivery was late and box crushed", "Fast delivery, love it", "Cracked, poor quality"],
            "rating": [1, 5, 2],
            "sentiment": ["Negative", "Positive", "Negative"],
        }
    )
    table = top_complaints(df)
    counts = dict(zip(table["theme"], table["mentions"]))
    assert counts == {"Shipping & Delivery": 1, "Product Quality": 1}


# ---------- ai_service ----------

@pytest.fixture
def reviews():
    return add_sentiment(load_reviews()["dataframe"])


def test_no_api_key_gives_friendly_error(monkeypatch, reviews):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    assert not ai_service.is_ai_configured()
    with pytest.raises(ai_service.AIServiceError, match="AI_API_KEY"):
        ai_service.generate_review_summary(reviews)


def test_model_name_is_configurable(monkeypatch):
    monkeypatch.delenv("AI_MODEL", raising=False)
    assert ai_service.get_model_name() == ai_service.DEFAULT_MODEL
    monkeypatch.setenv("AI_MODEL", "my-model")
    assert ai_service.get_model_name() == "my-model"


def test_summary_single_request_and_citation_check(monkeypatch, reviews):
    prompts = []

    def fake_call(system, prompt, max_tokens=2000):
        prompts.append(prompt)
        return "### Overall Summary\nMixed.\n\n### Key Issues\n- **Delivery** - recurring. [R003, R012, R999]"

    monkeypatch.setattr(ai_service, "_call_model", fake_call)
    result = ai_service.generate_review_summary(reviews, "all products")
    assert len(prompts) == 1 and result["chunks"] == 1
    assert "[R001]" in prompts[0]  # real review text is sent with IDs
    assert result["cited_ids"] == ["R003", "R012"]
    assert result["unknown_ids"] == ["R999"]


def test_large_dataset_is_chunked(monkeypatch, reviews):
    calls = []
    monkeypatch.setattr(ai_service, "CHUNK_CHAR_LIMIT", 800)
    monkeypatch.setattr(
        ai_service, "_call_model", lambda system, prompt, max_tokens=2000: calls.append(prompt) or "- note [R001]"
    )
    result = ai_service.generate_review_summary(reviews)
    assert result["chunks"] > 1
    assert len(calls) == result["chunks"] + 1  # one per chunk + one combine step
    assert "<batch_notes>" in calls[-1]


def test_api_errors_become_friendly(monkeypatch, reviews):
    import anthropic
    import httpx2

    monkeypatch.setenv("AI_API_KEY", "test-key-not-real")
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def raise_connection_error(self, **kwargs):
        raise anthropic.APIConnectionError(request=request)

    monkeypatch.setattr(anthropic.resources.Messages, "create", raise_connection_error)
    with pytest.raises(ai_service.AIServiceError, match="Could not reach"):
        ai_service.generate_review_summary(reviews)

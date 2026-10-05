"""Data loading, sentiment and Hindi/Hinglish support."""

import io

import pandas as pd
import pytest

from data_loader import load_reviews, validate_reviews
from sentiment import add_sentiment, detect_language, label_from_compound, score_text


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
    missing = tmp_path / "nope.csv"
    load_reviews(missing)
    assert not missing.exists()


def test_demo_dataset_is_valid(demo_reviews):
    assert len(demo_reviews) > 200
    assert {"sentiment", "sentiment_score", "language"} <= set(demo_reviews.columns)


# ---------- sentiment ----------

def test_sentiment_labels():
    assert label_from_compound(0.5) == "Positive"
    assert label_from_compound(-0.5) == "Negative"
    assert label_from_compound(0.0) == "Neutral"
    df = add_sentiment(pd.DataFrame({"review_text": ["I love it, excellent!", "Terrible, awful product."]}))
    assert df["sentiment"].tolist() == ["Positive", "Negative"]


def test_review_vocabulary_extension():
    assert label_from_compound(score_text("Bluetooth keeps disconnecting every few minutes.")) == "Negative"
    # "customer support" is a topic, not praise.
    assert score_text("Customer support did not respond to my emails.") < 0.05


# ---------- Hindi / Hinglish ----------

@pytest.mark.parametrize(
    "text, expected_language",
    [
        ("Product mast hai but battery bahut jaldi khatam ho jati hai.", "Hinglish"),
        ("Sound quality acchi hai but delivery late thi.", "Hinglish"),
        ("Phone ka camera awesome hai.", "Hinglish"),
        ("बैटरी बहुत जल्दी खत्म हो जाती है।", "Hindi"),
        ("The sound quality is great and delivery was fast.", "English"),
    ],
)
def test_language_detection(text, expected_language):
    assert detect_language(text) == expected_language


def test_hinglish_sentiment():
    assert label_from_compound(score_text("Phone ka camera awesome hai.")) == "Positive"
    assert label_from_compound(score_text("Product accha nahi hai.")) == "Negative"
    assert label_from_compound(score_text("Kuch kaam nahi karta, bilkul bekar.")) == "Negative"
    assert label_from_compound(score_text("Customer care ne koi jawab nahi diya.")) == "Negative"
    # Mixed review that ends with a complaint is not scored as positive.
    assert score_text("Product mast hai but battery bahut jaldi khatam ho jati hai.") < 0.05


def test_hindi_sentiment():
    assert label_from_compound(score_text("आवाज़ की गुणवत्ता बहुत अच्छी है।")) == "Positive"
    assert label_from_compound(score_text("बैटरी बहुत जल्दी खत्म हो जाती है।")) == "Negative"
    assert label_from_compound(score_text("प्रोडक्ट अच्छा नहीं है।")) == "Negative"


def test_language_column_added(demo_reviews):
    assert set(demo_reviews["language"]) <= {"English", "Hinglish", "Hindi"}
    assert (demo_reviews["language"] == "Hinglish").sum() > 0

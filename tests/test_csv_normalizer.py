"""Multi-file CSV upload: column detection, normalisation, cleaning, duplicates and combining."""

import time

import numpy as np
import pandas as pd
import pytest

import app
import csv_normalizer as cn
from conftest import DEMO_CSV, ROOT
from data_loader import REQUIRED_COLUMNS

SAMPLE_CSV = ROOT / "data" / "sample_reviews.csv"

ALT_CSV = (b"id,product,review,stars,date\n"
           b"1,Phone X,The battery life is excellent and the camera is sharp.,5,2026-03-25\n"
           b"2,Phone X,Screen cracked within a week of normal use.,1,2026-03-26\n"
           b"3,Speaker Mini,Loud and clear sound for such a small speaker.,4,2026-03-27\n")


def process(data: bytes, name: str = "file.csv", **kwargs) -> cn.FileResult:
    return cn.process_file(data, name, **kwargs)


# TEST 1: standard CSV
def test_standard_csv():
    result = process(SAMPLE_CSV.read_bytes(), "sample_reviews.csv")
    assert result.status == "ok" and len(result.dataframe) == 24
    assert result.detection.mapping == {c: c for c in REQUIRED_COLUMNS}
    assert list(result.dataframe.columns[:5]) == REQUIRED_COLUMNS
    assert result.dataframe["review_date"].dtype.kind == "M"


# TEST 2: alternative names, including "review" as the text and "stars" as the rating
def test_alternative_csv():
    result = process(ALT_CSV, "amazon_reviews.csv")
    assert result.status == "ok"
    assert result.detection.mapping == {"review_id": "id", "product_name": "product", "rating": "stars",
                                        "review_text": "review", "review_date": "date"}
    df = result.dataframe
    assert df["rating"].tolist() == [5.0, 1.0, 4.0]
    assert df["review_text"].iloc[0].startswith("The battery life")
    assert (df["source_file"] == "amazon_reviews.csv").all()


# TEST 3: different capitalisation and spacing
@pytest.mark.parametrize("header", [
    "Review ID,Product Name,Rating,Review Text,Review Date",
    "REVIEW_ID,PRODUCT_NAME,RATING,REVIEW_TEXT,REVIEW_DATE",
    "ReviewID,ProductName,Rating,ReviewText,ReviewDate",
    " review-id , product-name , rating , review-text , review-date ",
])
def test_capitalisation(header):
    data = (header + "\nA1,Speaker,4,Loud and clear sound but the bass is weak.,2026-03-25\n").encode()
    result = process(data)
    assert result.status == "ok", result.reason
    assert result.dataframe.iloc[0][REQUIRED_COLUMNS].tolist()[:4] == ["A1", "Speaker", 4.0,
                                                                       "Loud and clear sound but the bass is weak."]


# TEST 4: several files with different schemas are combined
def test_multiple_files_different_schemas():
    files = {
        "amazon_reviews.csv": ALT_CSV,
        "flipkart_reviews.csv": (b"review_id,product_name,review_text,rating,review_date\n"
                                 b"F1,Phone X,Good phone but it heats up while gaming.,3,2026-04-01\n"
                                 b"F2,Laptop Pro,Fast and light. Love the keyboard.,5,2026-04-02\n"),
        "customer_reviews.csv": (b"reviewID,product,feedback,score,created_at\n"
                                 b"C1,Sony TV,Picture quality is stunning and setup was easy,9,1711324800\n"
                                 b"C2,Sony TV,Remote stopped working after two days sadly,3,1711411200\n"),
    }
    results = [process(data, name) for name, data in files.items()]
    assert [r.status for r in results] == ["ok", "ok", "ok"]
    combined = cn.combine_uploaded_files(results)
    assert len(combined.reviews) == 7
    assert sorted(combined.reviews["product_name"].unique()) == ["Laptop Pro", "Phone X", "Sony TV", "Speaker Mini"]
    assert list(combined.reviews.columns[:5]) == REQUIRED_COLUMNS
    # The third file used a 1-10 scale and Unix timestamps.
    tv = combined.reviews[combined.reviews["product_name"] == "Sony TV"]
    assert tv["rating"].between(1, 5).all() and tv["review_date"].dt.year.eq(2024).all()


# TEST 5: one valid and one invalid file: the valid one is still used
def test_valid_and_invalid_file():
    good = process(ALT_CSV, "good.csv")
    bad = process(b"col_a,col_b\nhello,123\nworld,456\n", "bad.csv")
    assert good.status == "ok"
    assert bad.status == "needs_mapping" and "Review Text" in bad.reason
    combined = cn.combine_uploaded_files([good, bad])
    assert len(combined.reviews) == 3


# TEST 6: duplicate review IDs
def test_duplicate_ids():
    first = process(ALT_CSV, "a.csv")
    copy = process(ALT_CSV, "b.csv")  # the same three reviews again (exact duplicates)
    clash = process(b"id,product,review,stars,date\n1,Phone X,A completely different review text here.,2,2026-03-28\n",
                    "c.csv")
    kept = cn.combine_uploaded_files([first, copy, clash])
    assert len(kept.reviews) == 7                       # keep all by default: nothing deleted
    assert kept.exact_duplicates == 3 and kept.conflicting_ids == 1
    assert "1~2" in set(kept.reviews["review_id"])      # different content, same ID: kept and renamed
    removed = cn.combine_uploaded_files([first, copy, clash], remove_exact_duplicates=True)
    assert len(removed.reviews) == 4 and removed.removed_duplicates == 3
    assert removed.reviews["review_id"].is_unique


# TEST 7: missing dates are reported and those rows left out, without crashing
def test_missing_dates():
    data = (b"id,product,review,stars,date\n"
            b"1,Phone X,The battery life is excellent and lasts two days.,5,2026-03-25\n"
            b"2,Phone X,Screen cracked within a week of normal use.,1,\n"
            b"3,Phone X,Camera is decent in daylight but poor at night.,3,not a date\n")
    result = process(data)
    assert result.status == "ok" and len(result.dataframe) == 1 and result.missing_dates == 2


def test_whole_file_without_dates_needs_mapping_then_today():
    data = b"id,product,review,stars\n1,Phone X,The battery life is excellent and lasts two days.,5\n"
    result = process(data)
    assert result.status == "needs_mapping" and result.detection.missing == ["review_date"]
    mapping = {**result.detection.mapping, "review_date": cn.DATE_TODAY}
    fixed = process(data, mapping=mapping)
    assert fixed.status == "ok" and fixed.dataframe["review_date"].iloc[0] == pd.Timestamp.today().normalize()


# TEST 8: missing review text
def test_missing_review_text():
    data = (b"id,product,review,stars,date\n"
            b"1,Phone X,The battery life is excellent and lasts two days.,5,2026-03-25\n"
            b"2,Phone X,,1,2026-03-26\n"
            b"3,Phone X,   ,2,2026-03-27\n"
            b",,,,\n")
    result = process(data)
    assert result.status == "ok" and len(result.dataframe) == 1
    assert any("empty review text" in w for w in result.warnings)


# TEST 9: ratings stored as strings
@pytest.mark.parametrize("raw, expected", [(5, 5.0), ("5", 5.0), ("5 stars", 5.0), ("4/5", 4.0), ("4.5", 4.5),
                                           ("8 out of 10", 4.0), ("★★★", 3.0), ("3,5", 3.5), ("", np.nan),
                                           ("great", np.nan)])
def test_parse_rating_value(raw, expected):
    value = cn.parse_rating_value(raw)
    assert (np.isnan(value) and np.isnan(expected)) or value == expected


def test_string_ratings_in_file():
    data = (b"id,product,review,stars,date\n"
            b"1,Phone X,The battery life is excellent and lasts two days.,5 stars,2026-03-25\n"
            b"2,Phone X,Screen cracked within a week of normal use.,1/5,2026-03-26\n"
            b"3,Phone X,Camera is decent in daylight but poor at night.,\"3\",2026-03-27\n")
    result = process(data)
    assert result.status == "ok" and result.dataframe["rating"].tolist() == [5.0, 1.0, 3.0]


def test_ten_point_scale_can_be_left_unconverted():
    data = (b"id,product,review,score,date\n"
            b"1,TV,Picture quality is stunning and setup was easy.,9,2026-03-25\n"
            b"2,TV,Remote stopped working after two days sadly.,3,2026-03-26\n")
    converted = process(data)
    assert converted.scale == "1-10" and converted.dataframe["rating"].between(1, 5).all()
    raw = process(data, convert_rating_scale=False)
    assert len(raw.dataframe) == 1  # the 9 is out of range for 1-5, so it is removed as invalid


# TEST 10: empty CSV
@pytest.mark.parametrize("data, reason", [(b"", "empty"), (b"   \n", "empty"),
                                          (b"id,product,review,stars,date\n", "no rows")])
def test_empty_csv(data, reason):
    result = process(data, "empty.csv")
    assert result.status == "error" and reason in result.reason


# TEST 11: large CSV
def test_large_csv():
    rows = 50_000
    frame = pd.DataFrame({
        "Review ID": [f"L{i}" for i in range(rows)],
        "Product": np.random.default_rng(1).choice(["Phone X", "Laptop Pro", "Sony TV"], rows),
        "Stars": np.random.default_rng(2).integers(1, 6, rows),
        "Comment": ["Decent product overall, battery could be better though."] * rows,
        "Date": pd.date_range("2025-01-01", periods=rows, freq="10min").strftime("%Y-%m-%d %H:%M"),
    })
    start = time.perf_counter()
    result = process(frame.to_csv(index=False).encode(), "large.csv")
    assert result.status == "ok" and len(result.dataframe) == rows
    assert time.perf_counter() - start < 60


# TEST 12: Hindi / Hinglish text, and a non-UTF-8 file
def test_hindi_and_hinglish_text():
    data = ("id,product,review,stars,date\n"
            "1,Phone X,आवाज़ की गुणवत्ता बहुत अच्छी है और बैटरी भी बढ़िया है।,5,2026-03-25\n"
            "2,Phone X,Battery bahut jaldi khatam ho jati hai yaar.,2,2026-03-26\n").encode("utf-8")
    result = process(data)
    assert result.status == "ok"
    assert result.dataframe["review_text"].iloc[0].startswith("आवाज़ की गुणवत्ता")


def test_windows_encoded_file():
    data = "id,product,review,stars,date\n1,Café Maker,Great café-style coffee – really smooth.,5,2026-03-25\n"
    result = process(data.encode("cp1252"))
    assert result.status == "ok" and result.dataframe["product_name"].iloc[0] == "Café Maker"


def test_semicolon_separated_file():
    data = b"id;product;review;stars;date\n1;Phone X;The battery life is excellent and lasts two days.;5;2026-03-25\n"
    assert process(data).status == "ok"


# Detection safety
def test_content_detection_and_no_risky_guesses():
    data = (b"col_a,col_b,col_c,col_d\n"
            b"foo,Amazing headphones with great noise cancelling and comfort,5,2026-01-02\n"
            b"bar,Terrible build quality and the hinge broke quickly,1,2026-01-03\n")
    detection = cn.detect_column_mapping(cn.read_csv_bytes(data))
    assert detection.mapping["review_text"] == "col_b" and detection.methods["review_text"] == "content"
    assert detection.mapping["rating"] == "col_c" and detection.mapping["review_date"] == "col_d"
    assert detection.missing == ["product_name"]  # never guessed from values alone


def test_named_column_with_wrong_values_is_not_used():
    data = (b"id,product,review,score,date\n"
            b"1,Phone X,The battery life is excellent and lasts two days.,excellent value,2026-03-25\n")
    detection = cn.detect_column_mapping(cn.read_csv_bytes(data))
    assert "rating" in detection.missing


def test_manual_mapping_checks():
    columns = ["a", "b", "c"]
    good = {"review_id": cn.AUTO_ID, "product_name": cn.FILE_AS_PRODUCT, "rating": cn.RATING_FROM_TEXT,
            "review_text": "b", "review_date": cn.DATE_TODAY}
    assert cn.check_manual_mapping(good, columns) == []
    assert cn.check_manual_mapping({**good, "review_text": cn.AUTO_ID}, columns)
    assert cn.check_manual_mapping({**good, "rating": "b"}, columns)  # same column twice
    assert cn.check_manual_mapping({**good, "review_date": None}, columns)


def test_manual_mapping_with_defaults():
    data = b"a,b\nfoo,Amazing headphones with great noise cancelling and comfort\n"
    mapping = {"review_id": cn.AUTO_ID, "product_name": cn.FILE_AS_PRODUCT, "rating": cn.RATING_FROM_TEXT,
               "review_text": "b", "review_date": cn.DATE_TODAY}
    result = process(data, "earbuds.csv", mapping=mapping)
    row = result.dataframe.iloc[0]
    assert result.status == "ok" and row["product_name"] == "earbuds" and row["review_id"] == "earbuds-1"
    assert 4 <= row["rating"] <= 5


# The combined upload runs through the existing analysis pipeline
def test_combined_upload_runs_the_full_analysis():
    results = [process(DEMO_CSV.read_bytes(), "demo.csv"), process(ALT_CSV, "amazon_reviews.csv")]
    combined = cn.combine_uploaded_files(results)
    analysis = app.analyse_upload.__wrapped__(combined.reviews, "Loaded from 2 files", "[]")
    df = analysis["dataframe"]
    assert len(df) == 298
    assert {"sentiment", "emotion", "risk_level", "language"} <= set(df.columns)
    assert set(df["product_name"]) >= {"Phone X", "Speaker Mini", "Nimbus ANC Headphones"}
    assert not analysis["mentions"].empty and set(analysis["mentions"]["review_id"]) <= set(df["review_id"])
    ctx = app.build_context(analysis, "Phone X", ())
    assert set(ctx.reviews["product_name"]) == {"Phone X"} and len(ctx.reviews) == 2


def test_product_names_with_numbers_are_not_ratings():
    # Regression: "iPhone 15" / "Samsung S24" once passed the rating check, so the product column was rejected.
    data = (b"id,product,review,stars,date\n"
            b"A1,iPhone 15,The battery life is excellent and the camera is sharp.,5,2026-08-02\n"
            b"A2,Samsung S24,Charging is slow and the phone heats up during calls.,2,2026-08-10\n")
    result = process(data)
    assert result.status == "ok" and result.detection.mapping["product_name"] == "product"
    assert not cn.looks_like_rating(pd.Series(["iPhone 15", "Samsung S24", "u1"]))
    assert cn.looks_like_rating(pd.Series(["5", "4 stars", "3/5", "★★"]))

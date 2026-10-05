"""Shared fixtures. Tests never call the real AI API."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_loader import load_reviews, validate_reviews  # noqa: E402
from sentiment import add_sentiment  # noqa: E402
from themes import extract_mentions  # noqa: E402

DEMO_CSV = ROOT / "data" / "demo_electronics_reviews.csv"


@pytest.fixture(scope="session")
def sample_reviews() -> pd.DataFrame:
    """The 24-review synthetic sample, with sentiment."""
    return add_sentiment(load_reviews()["dataframe"])


@pytest.fixture(scope="session")
def demo_reviews() -> pd.DataFrame:
    """The larger synthetic demo dataset, with sentiment."""
    df, _ = validate_reviews(pd.read_csv(DEMO_CSV, encoding="utf-8"))
    return add_sentiment(df)


@pytest.fixture(scope="session")
def demo_mentions(demo_reviews) -> pd.DataFrame:
    return extract_mentions(demo_reviews)


@pytest.fixture(autouse=True)
def no_real_api_key(monkeypatch):
    """Make sure no real key leaks into tests."""
    monkeypatch.delenv("AI_API_KEY", raising=False)

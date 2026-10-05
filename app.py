"""ProductPulse AI - AI-powered product review analysis dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

import ai_service
import styles
import views
from data_loader import DATA_DIR, load_reviews
from fake_risk import assess_reviews
from insights import priority_ranking
from sentiment import add_sentiment
from themes import extract_mentions
from trends import detect_spikes

DATASETS = {
    "Demo · Electronics (295 reviews)": DATA_DIR / "demo_electronics_reviews.csv",
    "Demo · Mixed products (24 reviews)": DATA_DIR / "sample_reviews.csv",
}
UPLOAD_OPTION = "Upload my own CSV"
ALL_PRODUCTS = "All products"
ASSETS = Path(__file__).resolve().parent / "assets"


@st.cache_data(show_spinner="Analysing reviews...")
def analyse_dataset(file_bytes: bytes | None, file_name: str | None, sample_path: str | None) -> dict:
    """Load a dataset and run the per-review analysis once (cached per file).

    Fake-review risk is computed on the whole dataset so that duplicate and burst
    detection can see every review, not just the filtered selection.
    """
    if file_bytes is not None:
        source = io.BytesIO(file_bytes)
        source.name = file_name
    else:
        source = Path(sample_path)
    loaded = load_reviews(source)
    if file_bytes is None and loaded["error"] is None:
        loaded["is_sample"] = True
        loaded["source_label"] = "Synthetic demo data"
    reviews = assess_reviews(add_sentiment(loaded["dataframe"]))
    loaded["dataframe"] = reviews
    loaded["mentions"] = extract_mentions(reviews)
    return loaded


def sidebar() -> tuple[dict, str, tuple]:
    """Sidebar: dataset choice, filters and AI status. Returns (analysis, product, date_range)."""
    with st.sidebar:
        st.caption("AI-powered product review analysis")
        choice = st.selectbox("Dataset", list(DATASETS) + [UPLOAD_OPTION], key="dataset_choice")
        uploaded = None
        if choice == UPLOAD_OPTION:
            uploaded = st.file_uploader("Reviews CSV", type="csv",
                                        help="Columns: review_id, product_name, rating, review_text, review_date (or date).")
            if uploaded is None:
                st.info("Upload a CSV to analyse your own reviews. Showing the demo data meanwhile.")
        sample = None if uploaded else str(DATASETS.get(choice, next(iter(DATASETS.values()))))
        analysis = analyse_dataset(uploaded.getvalue() if uploaded else None, uploaded.name if uploaded else None, sample)
        df = analysis["dataframe"]

        st.divider()
        dataset_key = choice + (uploaded.name if uploaded else "")
        products = [ALL_PRODUCTS] + sorted(df["product_name"].unique())
        product = st.selectbox("Product", products, key=f"product_{dataset_key}")
        lo, hi = df["review_date"].min().date(), df["review_date"].max().date()
        date_range = st.date_input("Review dates", (lo, hi), min_value=lo, max_value=hi, key=f"dates_{dataset_key}")

        st.divider()
        if ai_service.is_ai_configured():
            st.success(f"AI on · `{ai_service.get_model_name()}`", icon="✨")
        else:
            st.warning("AI off: set AI_API_KEY to enable summaries and Q&A. All other analysis works.", icon="🔑")
        st.caption("Sentiment: VADER + Hinglish/Hindi lexicon · Themes: keyword rules · Risk: text similarity & patterns")
    return analysis, product, date_range


def build_context(analysis: dict, product: str, date_range: tuple) -> views.Context:
    """Filter the analysed data to the sidebar selection and compute selection-level analysis."""
    df = analysis["dataframe"]
    if product != ALL_PRODUCTS:
        df = df[df["product_name"] == product]
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = (pd.Timestamp(d) for d in date_range)
        df = df[(df["review_date"] >= start) & (df["review_date"] < end + pd.Timedelta(days=1))]
    mentions = analysis["mentions"]
    mentions = mentions[mentions["review_id"].isin(df["review_id"])]
    subject = product if product != ALL_PRODUCTS else "all products"

    if df.empty:
        priority, spikes, date_label = pd.DataFrame(), {"status": "insufficient", "alerts": [], "message": ""}, "No dates"
    else:
        priority = priority_ranking(df, mentions)
        spikes = detect_spikes(df, mentions, "" if product == ALL_PRODUCTS else product)
        date_label = f"{df['review_date'].min():%d %b %Y} – {df['review_date'].max():%d %b %Y}"
    ctx = views.Context(reviews=df, mentions=mentions, priority=priority, spikes=spikes, subject=subject,
                        source_label=analysis["source_label"], is_sample=analysis["is_sample"], date_label=date_label)
    return views.build_insights(ctx)


def main() -> None:
    st.set_page_config(page_title="ProductPulse AI", page_icon=str(ASSETS / "icon.svg"), layout="wide",
                       initial_sidebar_state="expanded")
    st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))
    styles.inject_css()

    analysis, product, date_range = sidebar()
    ctx = build_context(analysis, product, date_range)

    pages = {
        "overview": st.Page(lambda: views.overview(ctx), title="Overview", icon="🏠", url_path="overview", default=True),
        "ai": st.Page(lambda: views.ai_insights(ctx), title="AI Insights", icon="✨", url_path="ai-insights"),
        "ask": st.Page(lambda: views.ask(ctx), title="Ask Your Reviews", icon="💬", url_path="ask"),
        "priority": st.Page(lambda: views.priority(ctx), title="Priority Issues", icon="🎯", url_path="priority-issues"),
        "trends": st.Page(lambda: views.trends(ctx), title="Trends & Alerts", icon="📈", url_path="trends"),
        "integrity": st.Page(lambda: views.integrity(ctx), title="Review Integrity", icon="🛡️", url_path="review-integrity"),
        "explorer": st.Page(lambda: views.explorer(ctx), title="Review Explorer", icon="🔎", url_path="explorer"),
    }
    views.PAGES = pages
    navigation = st.navigation({
        "Dashboard": [pages["overview"]],
        "AI analysis": [pages["ai"], pages["ask"]],
        "Issues": [pages["priority"], pages["trends"]],
        "Trust & data": [pages["integrity"], pages["explorer"]],
    }, position="top")

    if analysis["error"]:
        st.error(analysis["error"])
    if analysis["is_sample"]:
        st.caption("ℹ️ Demo mode: these reviews are **synthetic** and not real customer feedback. "
                   "Choose “Upload my own CSV” in the sidebar to analyse your own data.")
    for message in analysis["warnings"]:
        st.caption(f"Data cleaning: {message}")

    navigation.run()


if __name__ == "__main__":
    main()

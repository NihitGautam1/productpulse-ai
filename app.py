"""ProductPulse AI - AI-powered product review analysis dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import ai_service
import styles
import upload_ui
import video_reviews
import views
from data_loader import DATA_DIR, load_reviews
from emotions import add_emotions
from fake_risk import assess_reviews
from insights import priority_ranking
from sentiment import add_sentiment
from themes import extract_mentions
from trends import detect_spikes

DATASETS = {
    "Demo · Electronics (295 reviews)": DATA_DIR / "demo_electronics_reviews.csv",
    "Demo · Mixed products (24 reviews)": DATA_DIR / "sample_reviews.csv",
}
UPLOAD_OPTION = "Upload my own CSV files"
ALL_PRODUCTS = "All products"
ASSETS = Path(__file__).resolve().parent / "assets"


@st.cache_data(show_spinner="Analysing reviews...")
def analyse_dataset(file_bytes: bytes | None, file_name: str | None, sample_path: str | None,
                    videos_json: str = "[]") -> dict:
    """Load a dataset, add any video reviews, and run the per-review analysis once (cached per input).

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
    return run_pipeline(loaded, videos_json)


@st.cache_data(show_spinner="Analysing reviews...")
def analyse_upload(reviews: pd.DataFrame, source_label: str, videos_json: str = "[]") -> dict:
    """Run the analysis on reviews combined from uploaded CSV files (already normalised and cleaned)."""
    loaded = {"dataframe": reviews, "source_label": source_label, "is_sample": False, "warnings": [], "error": None}
    return run_pipeline(loaded, videos_json)


def run_pipeline(loaded: dict, videos_json: str) -> dict:
    """Add video reviews, then sentiment, fake-review risk, emotions and theme mentions."""
    reviews = loaded["dataframe"].assign(source=ai_service.TEXT_SOURCE, rating_estimated=False)
    videos = json.loads(videos_json)
    if videos:
        reviews = pd.concat([reviews, video_reviews.videos_to_frame(videos)], ignore_index=True)
        reviews = reviews.sort_values("review_date").reset_index(drop=True)
        loaded["source_label"] += f" + {len(videos)} video review(s)"
    if "source_file" in reviews.columns:
        reviews["source_file"] = reviews["source_file"].fillna("Video reviews")
    reviews = add_emotions(assess_reviews(add_sentiment(reviews)))
    loaded["dataframe"] = reviews
    loaded["mentions"] = extract_mentions(reviews)
    return loaded


def sidebar() -> tuple[dict, str, tuple, upload_ui.UploadState | None]:
    """Sidebar: dataset choice, filters and AI status. Returns (analysis, product, date_range, upload state)."""
    with st.sidebar:
        st.caption("AI-powered product review analysis")
        choice = st.selectbox("Dataset", list(DATASETS) + [UPLOAD_OPTION], key="dataset_choice")
        upload = None
        if choice == UPLOAD_OPTION:
            files = upload_ui.uploader()
            if files:
                upload = upload_ui.process_uploads(files)
                upload_ui.sidebar_status(upload)
            if upload is None or not upload.usable:
                st.info("Upload CSV files to analyse your own reviews. Showing the demo data meanwhile.")
        if "video_reviews" not in st.session_state:  # first run of this session: bring back saved videos
            st.session_state["video_reviews"] = video_reviews.load_saved()
        videos = st.session_state["video_reviews"]
        videos_json = json.dumps(video_reviews.analysis_fields(videos), default=str)
        if upload is not None and upload.usable:
            files_ok = [r.name for r in upload.results if r.status == "ok"]
            label = f"Loaded from {files_ok[0]}" if len(files_ok) == 1 else f"Loaded from {len(files_ok)} files"
            analysis = analyse_upload(upload.combined.reviews, label, videos_json)
        else:
            sample = str(DATASETS.get(choice, next(iter(DATASETS.values()))))
            analysis = analyse_dataset(None, None, sample, videos_json)
        df = analysis["dataframe"]

        st.divider()
        # Product names always come from the data, so the list changes whenever the uploaded files change.
        dataset_key = choice + (upload.signature if upload is not None and upload.usable else "")
        products = [ALL_PRODUCTS] + sorted(df["product_name"].unique())
        product = st.selectbox("Product", products, key=f"product_{dataset_key}")
        lo, hi = df["review_date"].min().date(), df["review_date"].max().date()
        # New data (a video, a mapped file, a removed duplicate) can widen the date range, so the date picker
        # restarts with the full range whenever the range of the data changes.
        dates_key = f"dates_{dataset_key}|{lo}|{hi}"
        date_range = st.date_input("Review dates", (lo, hi), min_value=lo, max_value=hi, key=dates_key)
        if videos:
            st.caption(f"🎬 Includes {len(videos)} video review(s)")

        st.divider()
        if ai_service.is_ai_configured():
            st.success(f"AI on · {ai_service.provider_label()}", icon="✨")
        else:
            st.warning("AI off: set AI_API_KEY to enable summaries and Q&A. All other analysis works.", icon="🔑")
        st.caption("Sentiment: VADER + Hinglish/Hindi lexicon · Themes: keyword rules · Risk: text similarity & patterns")
        if st.button("👋 Welcome tour", type="tertiary"):
            views.welcome()
    return analysis, product, date_range, upload


def build_context(analysis: dict, product: str, date_range: tuple) -> views.Context:
    """Filter the analysed data to the sidebar selection and compute selection-level analysis."""
    df = analysis["dataframe"]
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = (pd.Timestamp(d) for d in date_range)
        df = df[(df["review_date"] >= start) & (df["review_date"] < end + pd.Timedelta(days=1))]
    all_reviews = df  # every product in the date range, for product comparisons
    all_mentions = analysis["mentions"][analysis["mentions"]["review_id"].isin(all_reviews["review_id"])]
    if product != ALL_PRODUCTS:
        df = df[df["product_name"] == product]
    mentions = all_mentions[all_mentions["review_id"].isin(df["review_id"])]
    subject = product if product != ALL_PRODUCTS else "all products"

    if df.empty:
        priority, spikes, date_label = pd.DataFrame(), {"status": "insufficient", "alerts": [], "message": ""}, "No dates"
    else:
        priority = priority_ranking(df, mentions)
        spikes = detect_spikes(df, mentions, "" if product == ALL_PRODUCTS else product)
        date_label = f"{df['review_date'].min():%d %b %Y} – {df['review_date'].max():%d %b %Y}"
    ctx = views.Context(reviews=df, mentions=mentions, priority=priority, spikes=spikes, subject=subject,
                        source_label=analysis["source_label"], is_sample=analysis["is_sample"], date_label=date_label,
                        all_reviews=all_reviews, all_mentions=all_mentions)
    return views.build_insights(ctx)


def load_cloud_secrets() -> None:
    """On Streamlit Community Cloud the API key is set in the app's Secrets, not in a .env file.
    Copy AI_* secrets into the environment, where ai_service reads them (local .env settings win)."""
    try:
        secrets = {key: str(value) for key, value in st.secrets.items() if key.startswith("AI_")}
    except Exception:  # no secrets file: running locally
        return
    for key, value in secrets.items():
        os.environ.setdefault(key, value)


def main() -> None:
    load_cloud_secrets()
    st.set_page_config(page_title="ProductPulse AI", page_icon=str(ASSETS / "icon.svg"), layout="wide",
                       initial_sidebar_state="expanded")
    st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))
    styles.inject_css()

    analysis, product, date_range, upload = sidebar()
    ctx = build_context(analysis, product, date_range)

    pages = {
        "overview": st.Page(lambda: views.overview(ctx), title="Overview", icon="🏠", url_path="overview", default=True),
        "compare": st.Page(lambda: views.compare(ctx), title="Compare Products", icon="⚖️", url_path="compare"),
        "ai": st.Page(lambda: views.ai_insights(ctx), title="AI Insights", icon="✨", url_path="ai-insights"),
        "ask": st.Page(lambda: views.ask(ctx), title="Ask Your Reviews", icon="💬", url_path="ask"),
        "priority": st.Page(lambda: views.priority(ctx), title="Priority Issues", icon="🎯", url_path="priority-issues"),
        "wishlist": st.Page(lambda: views.wishlist_page(ctx), title="Feature Wishlist", icon="💡",
                            url_path="feature-wishlist"),
        "replies": st.Page(lambda: views.replies_page(ctx), title="Reply Studio", icon="✉️", url_path="reply-studio"),
        "trends": st.Page(lambda: views.trends(ctx), title="Trends & Alerts", icon="📈", url_path="trends"),
        "integrity": st.Page(lambda: views.integrity(ctx), title="Review Integrity", icon="🛡️", url_path="review-integrity"),
        "explorer": st.Page(lambda: views.explorer(ctx), title="Review Explorer", icon="🔎", url_path="explorer"),
        "videos": st.Page(lambda: views.videos(ctx), title="Video Reviews", icon="🎬", url_path="video-reviews"),
    }
    views.PAGES = pages
    navigation = st.navigation({
        "Dashboard": [pages["overview"], pages["compare"]],
        "AI analysis": [pages["ai"], pages["ask"]],
        "Issues": [pages["priority"], pages["trends"], pages["wishlist"], pages["replies"]],
        "Trust & data": [pages["integrity"], pages["explorer"], pages["videos"]],
    }, position="top")

    if upload is not None:
        upload_ui.render_panel(upload)
    if analysis["error"]:
        st.error(analysis["error"])
    if analysis["is_sample"]:
        st.caption("ℹ️ Demo mode: these reviews are **synthetic** and not real customer feedback. "
                   "Choose “Upload my own CSV files” in the sidebar to analyse your own data.")
    for message in analysis["warnings"]:
        st.caption(f"Data cleaning: {message}")

    if not st.session_state.get("welcome_shown"):
        st.session_state["welcome_shown"] = True
        views.welcome()
    navigation.run()


if __name__ == "__main__":
    main()

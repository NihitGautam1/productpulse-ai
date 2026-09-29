"""ProductPulse AI Streamlit dashboard. All numbers are computed from the reviews file."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_loader import DEFAULT_CSV_PATH, load_reviews
from sentiment import add_sentiment, sentiment_counts
from themes import top_complaints

APP_DIR = Path(__file__).resolve().parent


def check_imports() -> list[str]:
    """Confirm the libraries this app needs can be imported."""
    problems: list[str] = []
    for name in ("pandas", "matplotlib", "streamlit", "vaderSentiment"):
        try:
            __import__(name if name != "vaderSentiment" else "vaderSentiment.vaderSentiment")
        except ImportError:
            problems.append(name)
    return problems


def apply_page_style() -> None:
    st.set_page_config(
        page_title="ProductPulse AI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px;}
            h1 {letter-spacing: -0.03em;}
            .pulse-banner {
                background: #fff7ed;
                border: 1px solid #fdba74;
                color: #9a3412;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
            }
            .stMetric {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 0.4rem 0.8rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def draw_sentiment_chart(counts: pd.Series) -> None:
    colors = {"Positive": "#16a34a", "Neutral": "#64748b", "Negative": "#dc2626"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = list(counts.index)
    values = [int(v) for v in counts.values]
    bars = ax.bar(labels, values, color=[colors[label] for label in labels], width=0.62)
    ax.set_ylabel("Number of reviews")
    ax.set_title("Sentiment mix (calculated from VADER scores)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(values + [1])
    ax.set_ylim(0, ymax * 1.18)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.03,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def filter_table(df: pd.DataFrame, query: str, product: str, sentiment: str) -> pd.DataFrame:
    view = df.copy()
    if product != "All products":
        view = view[view["product_name"] == product]
    if sentiment != "All sentiments":
        view = view[view["sentiment"] == sentiment]
    needle = query.strip().lower()
    if needle:
        mask = (
            view["review_text"].str.lower().str.contains(needle, na=False)
            | view["product_name"].str.lower().str.contains(needle, na=False)
            | view["review_id"].str.lower().str.contains(needle, na=False)
        )
        view = view[mask]
    return view


def main() -> None:
    apply_page_style()

    missing = check_imports()
    if missing:
        st.error(
            "Missing Python packages: "
            + ", ".join(missing)
            + ". Open PowerShell in this folder and run: pip install -r requirements.txt"
        )
        st.stop()

    st.sidebar.title("ProductPulse AI")
    st.sidebar.caption("Customer review intelligence for sellers")
    csv_path = st.sidebar.text_input("Reviews CSV path", value=str(DEFAULT_CSV_PATH))
    st.sidebar.markdown(
        "Required columns: `review_id`, `product_name`, `rating`, `review_text`, `review_date`."
    )

    loaded = load_reviews(csv_path)
    df = add_sentiment(loaded["dataframe"])
    counts = sentiment_counts(df)
    complaints = top_complaints(df)

    if loaded["is_synthetic"]:
        st.markdown(
            f'<div class="pulse-banner"><strong>Synthetic data in use.</strong> '
            f"{loaded['source_label']}</div>",
            unsafe_allow_html=True,
        )
    if loaded["error"]:
        st.warning(f"Your CSV was not used: {loaded['error']}")
    for warning in loaded["warnings"]:
        st.info(warning)

    products = ["All products"] + sorted(df["product_name"].dropna().unique().tolist())
    product_filter = st.sidebar.selectbox("Product", products)
    sentiment_filter = st.sidebar.selectbox(
        "Sentiment",
        ["All sentiments", "Positive", "Neutral", "Negative"],
    )

    filtered = df if product_filter == "All products" else df[df["product_name"] == product_filter]
    filtered_counts = sentiment_counts(filtered)
    filtered_complaints = top_complaints(filtered)

    st.title("ProductPulse AI")
    st.subheader("Overview")
    st.write(
        "This dashboard reads seller reviews, scores each one with VADER sentiment, "
        "and calculates ratings, complaint themes, and table results from that file. "
        "Nothing on this page is a hardcoded demo number."
    )

    total_reviews = int(len(filtered))
    average_rating = float(filtered["rating"].mean()) if total_reviews else 0.0
    positive_share = (
        float(filtered_counts["Positive"] / total_reviews * 100) if total_reviews else 0.0
    )

        negative_share = (
        float(filtered_counts["Negative"] / total_reviews * 100)
        if total_reviews else 0.0
    )

    metric_data = [
        ("Total Reviews", f"{total_reviews}"),
        ("Average Rating", f"{average_rating:.2f} / 5"),
        ("Positive Reviews", f"{int(filtered_counts['Positive'])}"),
        ("Negative Reviews", f"{negative_share:.1f}%"),
    ]

    cols = st.columns(4)

    for col, (label, value) in zip(cols, metric_data):
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color: #ffffff;
                    color: #111827;
                    border: 1px solid #d1d5db;
                    border-radius: 12px;
                    padding: 22px 16px;
                    min-height: 105px;
                    text-align: center;
                ">
                    <div style="
                        color: #4b5563;
                        font-size: 15px;
                        font-weight: 600;
                        margin-bottom: 12px;
                    ">{label}</div>
                    <div style="
                        color: #111827;
                        font-size: 28px;
                        font-weight: 700;
                    ">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns((1.15, 1))
    with left:
        st.markdown("### Sentiment")
        if total_reviews:
            draw_sentiment_chart(filtered_counts)
        else:
            st.write("No reviews match the current filters.")
    with right:
        st.markdown("### Top complaints")
        st.caption("Themes counted from negative sentiment or 1–2 star reviews.")
        if filtered_complaints.empty:
            st.write("No complaint themes found in the current data.")
        else:
            st.dataframe(filtered_complaints, use_container_width=True, hide_index=True)

    st.markdown("### Searchable review table")
    query = st.text_input(
        "Search by review text, product name, or review ID",
        placeholder="Try: shipping, battery, blender",
    )
    table = filter_table(df, query, product_filter, sentiment_filter)
    display = table[
        [
            "review_id",
            "product_name",
            "rating",
            "sentiment",
            "sentiment_score",
            "review_date",
            "review_text",
        ]
    ].copy()
    display["review_date"] = display["review_date"].dt.strftime("%Y-%m-%d")
    display["sentiment_score"] = display["sentiment_score"].round(3)
    st.caption(f"{len(display)} review(s) shown")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.sidebar.markdown("---")
    st.sidebar.write(f"Python {sys.version.split()[0]}")
    st.sidebar.write("Runs locally on CPU. No API keys required.")


if __name__ == "__main__":
    main()

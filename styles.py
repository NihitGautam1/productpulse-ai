"""Visual styling and small HTML components for the dashboard.

Colours are semi-transparent or inherit the theme text colour so every component
works in both the light and dark themes defined in .streamlit/config.toml.
All review text is HTML-escaped before it is inserted into the page.
"""

from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

KIND_STYLES = {
    "praise": ("#2a78d6", "Praise"),
    "complaint": ("#e34948", "Complaint"),
    "trend": ("#eb6834", "Trend"),
    "overview": ("#7a7974", "Overview"),
    "isolated": ("#a3a29c", "Isolated"),
    "language": ("#1baf7a", "Language"),
}
SEVERITY_STYLES = {"High": ("#d03b3b", "▲ High"), "Medium": ("#e09a00", "● Medium"), "Info": ("#2a78d6", "ℹ Info")}
LEVEL_BADGE_COLORS = {"Critical": "red", "High": "orange", "Medium": "yellow", "Low": "gray"}
RISK_BADGE_COLORS = {"High": "red", "Medium": "yellow", "Low": "gray"}

CSS = """
<style>
.block-container {padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1280px;}
[data-testid="stMetric"] {padding: 0.9rem 1rem;}
[data-testid="stMetricLabel"] p {font-weight: 600; opacity: 0.75; font-size: 0.85rem;}
[data-testid="stSidebarNav"] {padding-top: 0.25rem;}
.pp-hero {display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap;
          margin-bottom: 0.75rem;}
.pp-eyebrow {text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.75rem; font-weight: 600;
             color: #2a78d6; margin-bottom: 0.15rem;}
.pp-title {font-size: 2rem; font-weight: 700; line-height: 1.15; margin: 0; letter-spacing: -0.02em;}
.pp-subtitle {opacity: 0.7; margin-top: 0.35rem; font-size: 0.98rem;}
.pp-chips {display: flex; gap: 0.4rem; flex-wrap: wrap;}
.pp-chip {font-size: 0.78rem; padding: 0.2rem 0.6rem; border-radius: 999px;
          border: 1px solid rgba(128,128,128,0.35); opacity: 0.85; white-space: nowrap;}
.pp-card {border: 1px solid rgba(128,128,128,0.25); border-left: 4px solid var(--accent); border-radius: 0.6rem;
          padding: 0.7rem 0.9rem; margin-bottom: 0.6rem; background: rgba(128,128,128,0.04);}
.pp-card-label {font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
                color: var(--accent); margin-bottom: 0.2rem;}
.pp-card-body {font-size: 0.95rem; line-height: 1.45;}
.pp-card-meta {font-size: 0.78rem; opacity: 0.65; margin-top: 0.35rem;}
.pp-quote {border-left: 3px solid rgba(128,128,128,0.4); padding: 0.35rem 0 0.35rem 0.8rem; margin: 0.45rem 0;}
.pp-quote-text {font-size: 0.93rem; line-height: 1.45;}
.pp-quote-text mark {background: rgba(235,104,52,0.22); color: inherit; padding: 0 0.15rem; border-radius: 3px;}
.pp-quote-meta {font-size: 0.76rem; opacity: 0.65; margin-top: 0.2rem;}
.pp-section-title {font-size: 1.08rem; font-weight: 650; letter-spacing: -0.01em; margin: 0.1rem 0 0.35rem 0;}
.pp-section-caption {opacity: 0.68; font-size: 0.86rem; margin-top: -0.2rem; margin-bottom: 0.6rem;}
.pp-issue {display: flex; justify-content: space-between; align-items: center; gap: 0.6rem;
           padding: 0.55rem 0; border-bottom: 1px solid rgba(128,128,128,0.18);}
.pp-issue:last-child {border-bottom: none;}
.pp-issue-name {font-weight: 600;}
.pp-issue-meta {font-size: 0.8rem; opacity: 0.65;}
.pp-level {font-size: 0.75rem; font-weight: 700; padding: 0.15rem 0.55rem; border-radius: 999px; color: #fff;
           white-space: nowrap;}
</style>
"""


def inject_css() -> None:
    """Add the dashboard's custom CSS (call once per run)."""
    st.markdown(CSS, unsafe_allow_html=True)


def md_to_html(text: str) -> str:
    """Escape text, then convert **bold** markers into <strong> tags."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(str(text)))


def page_header(eyebrow: str, title: str, subtitle: str = "", chips: list[str] | None = None) -> None:
    """Large page title with a small label above it and optional info chips on the right."""
    chip_html = "".join(f'<span class="pp-chip">{html.escape(c)}</span>' for c in chips or [])
    st.markdown(
        f'<div class="pp-hero"><div><div class="pp-eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 class="pp-title">{html.escape(title)}</h1>'
        + (f'<div class="pp-subtitle">{html.escape(subtitle)}</div>' if subtitle else "")
        + f'</div><div class="pp-chips">{chip_html}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    """Section heading with an optional one-line explanation."""
    st.markdown(f'<div class="pp-section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="pp-section-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)


def insight_card(text: str, kind: str, evidence: list[str] | None = None) -> None:
    """Coloured card for one insight, with its evidence review IDs."""
    color, label = KIND_STYLES.get(kind, ("#7a7974", kind.title()))
    meta = f'<div class="pp-card-meta">Evidence: {html.escape(", ".join(evidence))}</div>' if evidence else ""
    st.markdown(
        f'<div class="pp-card" style="--accent:{color}"><div class="pp-card-label">{label}</div>'
        f'<div class="pp-card-body">{md_to_html(text)}</div>{meta}</div>',
        unsafe_allow_html=True,
    )


def alert_card(alert: dict) -> None:
    """Card for one spike alert."""
    color, label = SEVERITY_STYLES.get(alert["severity"], ("#7a7974", alert["severity"]))
    evidence = alert.get("evidence") or []
    meta = f'<div class="pp-card-meta">Recent examples: {html.escape(", ".join(evidence))}</div>' if evidence else ""
    st.markdown(
        f'<div class="pp-card" style="--accent:{color}"><div class="pp-card-label">{label} alert</div>'
        f'<div class="pp-card-body"><strong>{html.escape(alert["title"])}</strong><br>{html.escape(alert["detail"])}</div>'
        f"{meta}</div>",
        unsafe_allow_html=True,
    )


def quote(row: pd.Series, highlight: str | None = None) -> None:
    """Show a real review as a quote, optionally highlighting the clause that supports an insight."""
    text = html.escape(str(row["review_text"]))
    if highlight:
        escaped = html.escape(str(highlight))
        if escaped and escaped in text:
            text = text.replace(escaped, f"<mark>{escaped}</mark>", 1)
    stars = "★" * int(row["rating"]) + "☆" * (5 - int(row["rating"]))
    date = pd.Timestamp(row["review_date"]).strftime("%d %b %Y")
    meta = f"{html.escape(str(row['review_id']))} · {stars} · {date} · {html.escape(str(row['product_name']))}"
    st.markdown(
        f'<div class="pp-quote"><div class="pp-quote-text">“{text}”</div><div class="pp-quote-meta">{meta}</div></div>',
        unsafe_allow_html=True,
    )


def issue_row(theme: str, level: str, detail: str, color: str) -> str:
    """HTML for one row in a compact issue list."""
    text_color = "#1a1a19" if level == "Medium" else "#ffffff"  # dark text on the yellow badge for contrast
    return (f'<div class="pp-issue"><div><div class="pp-issue-name">{html.escape(theme)}</div>'
            f'<div class="pp-issue-meta">{html.escape(detail)}</div></div>'
            f'<span class="pp-level" style="background:{color};color:{text_color}">{html.escape(level)}</span></div>')


def html_block(content: str) -> None:
    """Render trusted HTML built by the helpers above."""
    st.markdown(content, unsafe_allow_html=True)

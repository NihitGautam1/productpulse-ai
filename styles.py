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
SENTIMENT_ACCENTS = {"Positive": "#2a78d6", "Neutral": "#8a8983", "Negative": "#e34948"}
QUOTE_CHARS = 420  # long texts (video transcripts) are shown as an excerpt around the highlighted part

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

/* Motion: gentle fade-in for cards, disabled for people who prefer reduced motion. */
@keyframes pp-fade {from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: none;}}
.pp-card, .pp-rcard, .pp-vcard, .pp-banner, .pp-vs {animation: pp-fade 0.45s ease-out both;}
@media (prefers-reduced-motion: reduce) {
  .pp-card, .pp-rcard, .pp-vcard, .pp-banner, .pp-vs, .pp-ring::before, .pp-count {animation: none !important;}
}

/* Hero banner with an animated health ring. */
@property --p {syntax: '<number>'; inherits: true; initial-value: 0;}
@property --n {syntax: '<integer>'; inherits: false; initial-value: 0;}
@keyframes pp-fill {from {--p: 0;}}
@keyframes pp-count {from {--n: 0;}}
.pp-banner {display: flex; gap: 1.6rem; align-items: center; flex-wrap: wrap; padding: 1.3rem 1.5rem;
            border-radius: 1rem; margin: 0.2rem 0 1rem 0; border: 1px solid rgba(42,120,214,0.28);
            background: linear-gradient(120deg, rgba(42,120,214,0.16), rgba(27,175,122,0.08) 55%, rgba(235,104,52,0.12));}
.pp-ring {position: relative; width: 128px; height: 128px; flex: none; display: grid; place-items: center;}
.pp-ring::before {content: ""; position: absolute; inset: 0; border-radius: 50%;
                  background: conic-gradient(var(--ring) calc(var(--p) * 1%), rgba(128,128,128,0.2) 0);
                  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 13px), #000 calc(100% - 12px));
                  mask: radial-gradient(farthest-side, transparent calc(100% - 13px), #000 calc(100% - 12px));
                  animation: pp-fill 1.4s cubic-bezier(.2,.8,.2,1) both;}
.pp-ring-value {text-align: center; line-height: 1;}
.pp-count {font-size: 2.3rem; font-weight: 750; letter-spacing: -0.03em; counter-reset: n var(--n);
           animation: pp-count 1.4s cubic-bezier(.2,.8,.2,1) both;}
.pp-count::after {content: counter(n);}
.pp-ring-label {font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.7; margin-top: 0.25rem;}
.pp-banner-text {flex: 1; min-width: 260px;}
.pp-headline {font-size: 1.32rem; font-weight: 650; line-height: 1.35; letter-spacing: -0.01em; margin: 0.15rem 0 0.6rem 0;}
.pp-headline strong {color: #2a78d6;}
.pp-headline strong.pp-bad {color: #e34948;}
.pp-pill {display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
          padding: 0.15rem 0.55rem; border-radius: 999px; background: rgba(42,120,214,0.16); color: #2a78d6;}

/* Review cards. */
.pp-rcard {border: 1px solid rgba(128,128,128,0.22); border-top: 3px solid var(--accent); border-radius: 0.7rem;
           padding: 0.75rem 0.9rem; margin-bottom: 0.75rem; background: rgba(128,128,128,0.035);
           transition: transform 0.15s ease, box-shadow 0.15s ease;}
.pp-rcard:hover, .pp-vcard:hover {transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.12);}
.pp-rcard-top {display: flex; gap: 0.35rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.4rem;}
.pp-stars {color: #eda100; letter-spacing: 0.05em; margin-right: 0.25rem;}
.pp-tag {font-size: 0.7rem; font-weight: 600; padding: 0.08rem 0.5rem; border-radius: 999px;
         border: 1px solid rgba(128,128,128,0.3); white-space: nowrap;}
.pp-tag-accent {border-color: var(--accent); color: var(--accent);}
.pp-rcard-text {font-size: 0.92rem; line-height: 1.5;}
.pp-rcard-text mark.pp-praise {background: rgba(42,120,214,0.18); color: inherit; border-radius: 3px; padding: 0 0.15rem;}
.pp-rcard-text mark.pp-complaint {background: rgba(227,73,72,0.18); color: inherit; border-radius: 3px; padding: 0 0.15rem;}
.pp-rcard-meta {font-size: 0.75rem; opacity: 0.62; margin-top: 0.45rem;}
.pp-moment {display: inline-block; font-size: 0.76rem; font-weight: 600; text-decoration: none !important;
            padding: 0.05rem 0.5rem; border-radius: 999px; background: rgba(232,123,164,0.18); color: #c2477a !important;
            margin-left: 0.3rem;}

/* Video cards. */
.pp-vcard {border: 1px solid rgba(128,128,128,0.22); border-radius: 0.8rem; overflow: hidden; margin-bottom: 0.4rem;
           background: rgba(128,128,128,0.035); transition: transform 0.15s ease, box-shadow 0.15s ease;}
.pp-thumb {position: relative; aspect-ratio: 16 / 9; background-size: cover; background-position: center;
           background-color: #1f1f1d; display: grid; place-items: center;}
.pp-thumb-upload {background-image: linear-gradient(135deg, #4a3aa7, #e87ba4);}
.pp-play {width: 52px; height: 52px; border-radius: 50%; display: grid; place-items: center; font-size: 1.3rem;
          background: rgba(0,0,0,0.55); color: #fff; text-decoration: none !important;}
.pp-vbadge {position: absolute; top: 0.5rem; left: 0.5rem; font-size: 0.68rem; font-weight: 700; color: #fff;
            padding: 0.1rem 0.5rem; border-radius: 999px; background: rgba(0,0,0,0.6);}
.pp-vsent {position: absolute; bottom: 0.5rem; right: 0.5rem; font-size: 0.7rem; font-weight: 700; color: #fff;
           padding: 0.1rem 0.55rem; border-radius: 999px; background: var(--accent);}
.pp-vbody {padding: 0.65rem 0.8rem 0.75rem 0.8rem;}
.pp-vtitle {font-weight: 650; font-size: 0.95rem; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2;
            -webkit-box-orient: vertical; overflow: hidden;}
.pp-vmeta {font-size: 0.76rem; opacity: 0.65; margin-top: 0.3rem;}

/* Word clouds. */
.pp-cloud {display: flex; flex-wrap: wrap; gap: 0.15rem 0.7rem; align-items: baseline; justify-content: center;
           padding: 0.6rem 0.2rem; line-height: 1.25;}
.pp-cloud span {color: var(--accent); font-weight: 650; letter-spacing: -0.01em; white-space: nowrap;}

/* Welcome tour tiles. */
.pp-tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.7rem; margin: 0.4rem 0 0.8rem 0;}
.pp-tile {border: 1px solid rgba(128,128,128,0.25); border-radius: 0.8rem; padding: 0.8rem 0.9rem;
          background: linear-gradient(160deg, rgba(42,120,214,0.10), transparent 75%); animation: pp-fade 0.5s ease-out both;}
.pp-tile-icon {font-size: 1.5rem;}
.pp-tile-title {font-weight: 700; margin: 0.25rem 0 0.15rem 0;}
.pp-tile-text {font-size: 0.84rem; opacity: 0.75; line-height: 1.4;}

/* Head-to-head comparison. */
.pp-vs {display: grid; grid-template-columns: 1fr auto 1fr; gap: 0.8rem; align-items: stretch; margin-bottom: 0.8rem;}
.pp-vs-side {border-radius: 0.9rem; padding: 1rem 1.1rem; border: 1px solid rgba(128,128,128,0.25);
             background: linear-gradient(160deg, color-mix(in srgb, var(--accent) 16%, transparent), transparent 70%);}
.pp-vs-name {font-weight: 700; font-size: 1.1rem; color: var(--accent); margin-bottom: 0.5rem;}
.pp-vs-row {display: flex; justify-content: space-between; padding: 0.28rem 0; border-bottom: 1px dashed rgba(128,128,128,0.2);
            font-size: 0.92rem;}
.pp-vs-row:last-child {border-bottom: none;}
.pp-vs-win {font-weight: 700;}
.pp-vs-win::after {content: " 🏆"; font-size: 0.8rem;}
.pp-vs-mid {display: grid; place-items: center; font-weight: 800; font-size: 1.1rem; opacity: 0.6;}
@media (max-width: 640px) {.pp-vs {grid-template-columns: 1fr;} .pp-vs-mid {display: none;}}
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


def stars(rating) -> str:
    rating = int(round(float(rating)))
    return "★" * rating + "☆" * (5 - rating)


def excerpt(text: str, focus: str | None, limit: int = QUOTE_CHARS) -> str:
    """Shorten long text (such as a video transcript) to a window around `focus`."""
    text = str(text)
    if len(text) <= limit:
        return text
    start = text.find(focus) if focus else -1
    if start < 0:
        return text[:limit].rsplit(" ", 1)[0] + " …"
    begin = max(0, start - limit // 3)
    end = min(len(text), begin + limit)
    return ("… " if begin else "") + text[begin:end] + (" …" if end < len(text) else "")


def moment_html(moment: tuple[str, str] | None) -> str:
    if not moment:
        return ""
    url, label = moment
    return f'<a class="pp-moment" href="{html.escape(url)}" target="_blank" rel="noopener">▶ Watch at {html.escape(label)}</a>'


def quote(row: pd.Series, highlight: str | None = None, moment: tuple[str, str] | None = None) -> None:
    """Show a real review as a quote, optionally highlighting the clause that supports an insight.

    `moment` is (url, "m:ss") for a video review: a link that plays the video where the clause is said.
    """
    text = html.escape(excerpt(row["review_text"], highlight))
    if highlight:
        escaped = html.escape(str(highlight))
        if escaped and escaped in text:
            text = text.replace(escaped, f"<mark>{escaped}</mark>", 1)
    date = pd.Timestamp(row["review_date"]).strftime("%d %b %Y")
    video = "🎬 " if str(row.get("source", "Text review")) != "Text review" else ""
    meta = (f"{video}{html.escape(str(row['review_id']))} · {stars(row['rating'])} · {date} · "
            f"{html.escape(str(row['product_name']))}{moment_html(moment)}")
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


def score_color(score: float) -> str:
    """Green for a healthy score, amber for middling, red for poor."""
    return "#1baf7a" if score >= 70 else "#eda100" if score >= 50 else "#e34948"


def hero(score: int, label: str, headline_text: str, chips: list[str]) -> None:
    """Dashboard banner: animated health ring plus a one-sentence headline (supports **bold**)."""
    chip_html = "".join(f'<span class="pp-chip">{html.escape(c)}</span>' for c in chips)
    st.markdown(
        f'<div class="pp-banner" style="--ring:{score_color(score)}">'
        f'<div class="pp-ring" style="--p:{int(score)}"><div class="pp-ring-value">'
        f'<div class="pp-count" style="--n:{int(score)}"></div><div class="pp-ring-label">Health</div></div></div>'
        f'<div class="pp-banner-text"><span class="pp-pill">{html.escape(label)}</span>'
        f'<div class="pp-headline">{re.sub(r"__(.+?)__", r"<strong class=pp-bad>\1</strong>", md_to_html(headline_text))}</div><div class="pp-chips">{chip_html}</div></div></div>',
        unsafe_allow_html=True,
    )


def _highlight_clauses(text: str, clauses: list[tuple[str, str]]) -> str:
    """Escape text and mark each (clause, polarity) found in it as praise or complaint."""
    escaped = html.escape(text)
    for clause, polarity in clauses:
        if polarity not in ("praise", "complaint"):
            continue
        target = html.escape(clause)
        if target and target in escaped:
            escaped = escaped.replace(target, f'<mark class="pp-{polarity}">{target}</mark>', 1)
    return escaped


def review_card(row: pd.Series, clauses: list[tuple[str, str]] | None = None,
                moment: tuple[str, str] | None = None, emotion=None) -> str:
    """HTML for one review as a card: stars, tags, text with praise/complaint highlights, and details.

    `emotion` is a function that turns an emotion name into its display label.
    """
    accent = SENTIMENT_ACCENTS.get(row.get("sentiment"), "#8a8983")
    clauses = clauses or []
    focus = next((c for c, p in clauses if p == "complaint"), clauses[0][0] if clauses else None)
    text = _highlight_clauses(excerpt(row["review_text"], focus), clauses)
    tags = [f'<span class="pp-tag pp-tag-accent">{html.escape(str(row.get("sentiment", "")))}</span>']
    if emotion and row.get("emotion") and row["emotion"] != "No clear emotion":
        tags.append(f'<span class="pp-tag">{html.escape(emotion(row["emotion"]))}</span>')
    if row.get("language") and row["language"] != "English":
        tags.append(f'<span class="pp-tag">{html.escape(row["language"])}</span>')
    source = str(row.get("source", "Text review"))
    if source != "Text review":
        tags.append(f'<span class="pp-tag">🎬 {html.escape(source)}</span>')
    if row.get("risk_level") == "High":
        tags.append('<span class="pp-tag" style="border-color:#d03b3b;color:#d03b3b">⚠ High risk</span>')
    estimated = " (est.)" if row.get("rating_estimated") is True else ""
    date = pd.Timestamp(row["review_date"]).strftime("%d %b %Y")
    return (f'<div class="pp-rcard" style="--accent:{accent}"><div class="pp-rcard-top">'
            f'<span class="pp-stars">{stars(row["rating"])}{estimated}</span>{"".join(tags)}</div>'
            f'<div class="pp-rcard-text">{text}</div>'
            f'<div class="pp-rcard-meta">{html.escape(str(row["review_id"]))} · {date} · '
            f'{html.escape(str(row["product_name"]))}{moment_html(moment)}</div></div>')


def video_card(video: dict, sentiment: str | None, thumbnail: str | None, views: str = "") -> str:
    """HTML card for a video review: thumbnail with play button, title, channel, views, date and rating."""
    accent = SENTIMENT_ACCENTS.get(sentiment, "#8a8983")
    url = html.escape(video.get("video_url") or "")
    play = (f'<a class="pp-play" href="{url}" target="_blank" rel="noopener">▶</a>' if url
            else '<span class="pp-play">🎬</span>')
    style = f' style="background-image:url({html.escape(thumbnail)})"' if thumbnail else ""
    badge = "YouTube" if video.get("video_id") else "Uploaded"
    sentiment_badge = f'<span class="pp-vsent">{html.escape(sentiment)}</span>' if sentiment else ""
    by = html.escape(video.get("channel") or video.get("transcript_method", ""))
    date = pd.Timestamp(video["review_date"]).strftime("%d %b %Y")
    estimated = " (est.)" if video.get("rating_estimated") else ""
    return (f'<div class="pp-vcard" style="--accent:{accent}">'
            f'<div class="pp-thumb{"" if thumbnail else " pp-thumb-upload"}"{style}>'
            f'<span class="pp-vbadge">{badge}</span>{play}{sentiment_badge}</div>'
            f'<div class="pp-vbody"><div class="pp-vtitle">{html.escape(video["video_title"])}</div>'
            f'<div class="pp-vmeta">{by}{" · " + html.escape(views) if views else ""} · {date}</div>'
            f'<div class="pp-vmeta"><span class="pp-stars">{stars(video["rating"])}</span>{estimated} · '
            f'{html.escape(video["product_name"])} · {html.escape(video["review_id"])}</div></div></div>')


def word_cloud(words: list[tuple[str, int]], color: str) -> None:
    """Words sized by how many reviews use them (largest = most common)."""
    if not words:
        st.caption("Not enough repeated words yet.")
        return
    top = max(n for _, n in words)
    low = min(n for _, n in words)
    spans = []
    for word, count in sorted(words, key=lambda item: item[0]):  # alphabetical, so big words are spread out
        weight = (count - low) / (top - low) if top > low else 1
        spans.append(f'<span style="font-size:{0.85 + weight * 1.45:.2f}rem;opacity:{0.55 + weight * 0.45:.2f}" '
                     f'title="{count} reviews">{html.escape(word)}</span>')
    st.markdown(f'<div class="pp-cloud" style="--accent:{color}">{"".join(spans)}</div>', unsafe_allow_html=True)


def tiles(items: list[tuple[str, str, str]]) -> None:
    """Grid of feature tiles: (icon, title, text)."""
    st.markdown('<div class="pp-tiles">' + "".join(
        f'<div class="pp-tile"><div class="pp-tile-icon">{icon}</div><div class="pp-tile-title">{html.escape(title)}</div>'
        f'<div class="pp-tile-text">{html.escape(text)}</div></div>' for icon, title, text in items) + "</div>",
        unsafe_allow_html=True)


def versus(names: tuple[str, str], rows: list[tuple[str, str, str, int]], colors: tuple[str, str]) -> None:
    """Two product panels side by side. rows: (metric, value A, value B, winner 0/1/-1)."""
    def side(index: int) -> str:
        lines = "".join(
            f'<div class="pp-vs-row"><span>{html.escape(metric)}</span>'
            f'<span class="{"pp-vs-win" if win == index else ""}">{html.escape(values[index])}</span></div>'
            for metric, *values, win in rows
        )
        return (f'<div class="pp-vs-side" style="--accent:{colors[index]}">'
                f'<div class="pp-vs-name">{html.escape(names[index])}</div>{lines}</div>')
    st.markdown(f'<div class="pp-vs">{side(0)}<div class="pp-vs-mid">VS</div>{side(1)}</div>', unsafe_allow_html=True)

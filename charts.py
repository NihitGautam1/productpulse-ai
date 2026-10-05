"""Plotly chart builders for the dashboard.

Colour rules (consistent across every chart):
- Sentiment is a polarity, so it uses a diverging pair: blue = positive, grey = neutral, red = negative.
- Priority levels use reserved status colours and are always shown with a text label.
- Themes over time use a fixed categorical order, so a theme keeps its colour when filters change.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

SENTIMENT_COLORS = {"Positive": "#2a78d6", "Neutral": "#a3a29c", "Negative": "#e34948"}
LEVEL_COLORS = {"Critical": "#d03b3b", "High": "#ec835a", "Medium": "#fab219", "Low": "#a3a29c"}
RISK_COLORS = {"High": "#d03b3b", "Medium": "#fab219", "Low": "#a3a29c"}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
THEME_ORDER = [
    "Connectivity", "Battery & Charging", "Shipping & Delivery", "Product Quality", "Customer Service",
    "Performance & Features", "Comfort & Materials", "Price & Value", "Design & Usability",
]


def theme_color(theme: str) -> str:
    """Stable colour per theme (fixed order, never re-assigned by rank)."""
    index = THEME_ORDER.index(theme) if theme in THEME_ORDER else len(THEME_ORDER)
    return CATEGORICAL[index % len(CATEGORICAL)]


def _layout(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font_size=13),
        bargap=0.35,
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def sentiment_bars(counts: pd.Series) -> go.Figure:
    """Horizontal bars: number of positive / neutral / negative reviews, with share labels."""
    total = max(int(counts.sum()), 1)
    order = ["Negative", "Neutral", "Positive"]  # Positive ends up on top
    values = [int(counts.get(label, 0)) for label in order]
    fig = go.Figure(go.Bar(
        x=values, y=order, orientation="h",
        marker=dict(color=[SENTIMENT_COLORS[l] for l in order], cornerradius=4),
        text=[f"{v}  ·  {v / total:.0%}" for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} reviews<extra></extra>",
    ))
    fig.update_xaxes(visible=False, range=[0, max(values + [1]) * 1.3])
    fig.update_yaxes(showgrid=False)
    return _layout(fig, height=200)


def rating_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of how many reviews gave each star rating."""
    counts = df["rating"].round().astype(int).value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    fig = go.Figure(go.Bar(
        x=[f"{s}★" for s in counts.index], y=counts.values,
        marker=dict(color="#2a78d6", cornerradius=4),
        text=counts.values, textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y} reviews<extra></extra>",
    ))
    fig.update_yaxes(visible=False, range=[0, max(counts.max(), 1) * 1.25])
    return _layout(fig, height=220)


def praise_vs_complaints(summary: pd.DataFrame) -> go.Figure:
    """Diverging bars per theme: complaints to the left (red), praise to the right (blue)."""
    data = summary.sort_values("complaints", ascending=True)
    fig = go.Figure()
    fig.add_bar(
        y=data["theme"], x=-data["complaints"], orientation="h", name="Complaints",
        marker=dict(color=SENTIMENT_COLORS["Negative"], cornerradius=4),
        customdata=data["complaints"], hovertemplate="%{y}<br>%{customdata} reviews complain<extra></extra>",
    )
    fig.add_bar(
        y=data["theme"], x=data["praise"], orientation="h", name="Praise",
        marker=dict(color=SENTIMENT_COLORS["Positive"], cornerradius=4),
        hovertemplate="%{y}<br>%{x} reviews praise<extra></extra>",
    )
    limit = max(data["complaints"].max(), data["praise"].max(), 1) * 1.1
    ticks = [round(limit * f) for f in (-1, -0.5, 0, 0.5, 1)]
    fig.update_layout(barmode="relative")
    fig.update_xaxes(range=[-limit, limit], tickvals=ticks, ticktext=[str(abs(t)) for t in ticks],
                     showgrid=True, gridcolor="rgba(128,128,128,0.15)", title_text="Reviews")
    fig.update_yaxes(showgrid=False)
    fig.add_vline(x=0, line_width=1, line_color="rgba(128,128,128,0.6)")
    return _layout(fig, height=max(260, 38 * len(data) + 60), legend=True)


def priority_scores(priority: pd.DataFrame) -> go.Figure:
    """Horizontal bars of priority score, coloured by level and labelled with the level name."""
    data = priority.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=data["score"], y=data["theme"], orientation="h",
        marker=dict(color=[LEVEL_COLORS[l] for l in data["level"]], cornerradius=4),
        text=[f"{lvl} · {s:.0f}" for lvl, s in zip(data["level"], data["score"])],
        textposition="outside", cliponaxis=False,
        customdata=data[["complaint_reviews", "share_pct"]],
        hovertemplate="%{y}<br>Score %{x:.1f}<br>%{customdata[0]} complaint reviews (%{customdata[1]}%)<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 115], title_text="Priority score (0-100)")
    fig.update_yaxes(showgrid=False)
    return _layout(fig, height=max(240, 38 * len(data) + 60))


def sentiment_volume(trends: pd.DataFrame) -> go.Figure:
    """Stacked bars per period: positive, neutral and negative review counts."""
    neutral = trends["reviews"] - trends["positive"] - trends["negative"]
    fig = go.Figure()
    for name, values in (("Negative", trends["negative"]), ("Neutral", neutral), ("Positive", trends["positive"])):
        fig.add_bar(x=trends["period"], y=values, name=name, marker=dict(color=SENTIMENT_COLORS[name]),
                    hovertemplate=f"%{{x|%d %b %Y}}<br>{name}: %{{y}}<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.2)
    fig.update_yaxes(title_text="Reviews")
    return _layout(fig, height=300, legend=True)


def line_chart(trends: pd.DataFrame, column: str, label: str, color: str, y_range: list | None = None,
               suffix: str = "") -> go.Figure:
    """Single-series line chart over time (one axis only)."""
    fig = go.Figure(go.Scatter(
        x=trends["period"], y=trends[column], mode="lines+markers", name=label, connectgaps=False,
        line=dict(color=color, width=2), marker=dict(size=8, line=dict(width=2, color="rgba(255,255,255,0.9)")),
        hovertemplate=f"%{{x|%d %b %Y}}<br>{label}: %{{y}}{suffix}<extra></extra>",
    ))
    fig.update_yaxes(title_text=label, range=y_range)
    fig.update_layout(hovermode="x unified")
    return _layout(fig, height=260)


def theme_trend_lines(theme_trends: pd.DataFrame, top: int = 5) -> go.Figure:
    """Complaint reviews per theme over time, for the most-mentioned themes."""
    fig = go.Figure()
    for theme in list(theme_trends.columns[:top]):
        fig.add_scatter(
            x=theme_trends.index, y=theme_trends[theme], mode="lines+markers", name=theme,
            line=dict(color=theme_color(theme), width=2), marker=dict(size=8),
            hovertemplate=f"{theme}: %{{y}}<extra></extra>",
        )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="Complaint reviews")
    return _layout(fig, height=320, legend=True)


def risk_levels(risk: pd.DataFrame) -> go.Figure:
    """Count of reviews per fake-review risk level."""
    order = ["Low", "Medium", "High"]
    counts = risk["risk_level"].value_counts().reindex(order, fill_value=0)
    fig = go.Figure(go.Bar(
        x=order, y=counts.values, marker=dict(color=[RISK_COLORS[o] for o in order], cornerradius=4),
        text=counts.values, textposition="outside", cliponaxis=False,
        hovertemplate="%{x} risk: %{y} reviews<extra></extra>",
    ))
    fig.update_yaxes(visible=False, range=[0, max(counts.max(), 1) * 1.25])
    return _layout(fig, height=220)


def language_mix(df: pd.DataFrame) -> go.Figure:
    """Reviews per detected language."""
    order = ["English", "Hinglish", "Hindi"]
    counts = df["language"].value_counts().reindex(order, fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.values, y=order, orientation="h", marker=dict(color=CATEGORICAL[:3], cornerradius=4),
        text=counts.values, textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} reviews<extra></extra>",
    ))
    fig.update_xaxes(visible=False, range=[0, max(counts.max(), 1) * 1.25])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _layout(fig, height=170)

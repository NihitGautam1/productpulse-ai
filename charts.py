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
COMPARE_COLORS = ["#2a78d6", "#eb6834"]  # product A, product B
VIDEO_COLOR = "#e87ba4"
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
        marker=dict(color=[SENTIMENT_COLORS[label] for label in order], cornerradius=4),
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
        marker=dict(color=[LEVEL_COLORS[level] for level in data["level"]], cornerradius=4),
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
               suffix: str = "", events: pd.DataFrame | None = None) -> go.Figure:
    """Single-series line chart over time (one axis only).

    `events` (columns: date, value, label) adds star markers, e.g. for video reviews on the timeline.
    """
    fig = go.Figure(go.Scatter(
        x=trends["period"], y=trends[column], mode="lines+markers", name=label, connectgaps=False,
        line=dict(color=color, width=2), marker=dict(size=8, line=dict(width=2, color="rgba(255,255,255,0.9)")),
        hovertemplate=f"%{{x|%d %b %Y}}<br>{label}: %{{y}}{suffix}<extra></extra>",
    ))
    if events is not None and not events.empty:
        fig.add_scatter(
            x=events["date"], y=events["value"], mode="markers", name="Video review",
            marker=dict(symbol="star", size=16, color=VIDEO_COLOR, line=dict(width=1, color="rgba(255,255,255,0.9)")),
            customdata=events["label"], hovertemplate="🎬 %{customdata}<br>%{x|%d %b %Y} · %{y}★<extra></extra>",
        )
    fig.update_yaxes(title_text=label, range=y_range)
    fig.update_layout(hovermode="closest" if events is not None and not events.empty else "x unified")
    return _layout(fig, height=260, legend=events is not None and not events.empty)


EMOTION_COLORS = {"Anger": "#d03b3b", "Frustration": "#eb6834", "Disappointment": "#a46ab8",
                  "Confusion": "#eda100", "Delight": "#1baf7a", "Satisfaction": "#2a78d6",
                  "No clear emotion": "#a3a29c"}


def emotion_bars(counts: pd.Series, labels: dict[str, str]) -> go.Figure:
    """Horizontal bars of reviews per emotion (emotions with no reviews are left out)."""
    data = counts[counts > 0].iloc[::-1]
    total = max(int(counts.sum()), 1)
    fig = go.Figure(go.Bar(
        x=data.values, y=[labels.get(e, e) for e in data.index], orientation="h",
        marker=dict(color=[EMOTION_COLORS.get(e, "#a3a29c") for e in data.index], cornerradius=4),
        text=[f"{v}  ·  {v / total:.0%}" for v in data.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} reviews<extra></extra>",
    ))
    fig.update_xaxes(visible=False, range=[0, max(data.max() if len(data) else 1, 1) * 1.35])
    fig.update_yaxes(showgrid=False)
    return _layout(fig, height=max(170, 34 * len(data) + 40))


def fix_impact(impact: pd.DataFrame) -> go.Figure:
    """Average rating now vs if each issue were fixed (a dumbbell per issue)."""
    data = impact.head(8).iloc[::-1]
    fig = go.Figure()
    for row in data.itertuples():
        fig.add_shape(type="line", x0=row.current_avg, x1=row.projected_avg, y0=row.theme, y1=row.theme,
                      line=dict(color="rgba(27,175,122,0.55)", width=6))
    fig.add_scatter(x=data["current_avg"], y=data["theme"], mode="markers", name="Now",
                    marker=dict(size=12, color="#a3a29c"), hovertemplate="Now: %{x:.2f}★<extra></extra>")
    fig.add_scatter(x=data["projected_avg"], y=data["theme"], mode="markers+text", name="If fixed",
                    marker=dict(size=14, color="#1baf7a"), text=[f"+{g:.2f}★" for g in data["gain"]],
                    textposition="middle right", cliponaxis=False,
                    customdata=data["affected"],
                    hovertemplate="If fixed: %{x:.2f}★<br>%{customdata} complaint reviews<extra></extra>")
    low = float(data["current_avg"].min()) if len(data) else 1
    high = float(data["projected_avg"].max()) if len(data) else 5
    fig.update_xaxes(range=[max(1, low - 0.15), min(5.4, high + 0.35)], title_text="Average rating (★)",
                     showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=False)
    return _layout(fig, height=max(220, 40 * len(data) + 70), legend=True)


def theme_map(summary: pd.DataFrame) -> go.Figure:
    """Bubble chart: how much each theme is discussed (size, height) and how customers feel about it (left-right)."""
    data = summary.copy()
    rated = data["praise"] + data["complaints"]
    data = data[rated > 0]
    rated = rated[rated > 0]
    data["net"] = ((data["praise"] - data["complaints"]) / rated * 100).round(0)
    data["mentions"] = data["praise"] + data["complaints"] + data["neutral"]
    size_ref = 2.0 * max(data["mentions"].max(), 1) / (58 ** 2)
    fig = go.Figure(go.Scatter(
        x=data["net"], y=data["mentions"], mode="markers+text", text=data["theme"], textposition="top center",
        marker=dict(size=data["mentions"], sizemode="area", sizeref=size_ref, sizemin=10,
                    color=data["net"], colorscale=[[0, SENTIMENT_COLORS["Negative"]], [0.5, "#c9c8c2"],
                                                   [1, SENTIMENT_COLORS["Positive"]]],
                    cmin=-100, cmax=100, opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.8)")),
        customdata=data[["praise", "complaints"]],
        hovertemplate="<b>%{text}</b><br>%{y} reviews mention it<br>%{customdata[0]} praise · "
                      "%{customdata[1]} complain<br>Net sentiment %{x:+.0f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="rgba(128,128,128,0.6)")
    top = max(data["mentions"].max(), 1) * 1.3
    for x, label, anchor in ((-100, "← Mostly complaints", "left"), (100, "Mostly praise →", "right")):
        fig.add_annotation(x=x, y=top, text=label, showarrow=False, xanchor=anchor, font=dict(size=12),
                           opacity=0.6)
    fig.update_xaxes(range=[-118, 118], title_text="Net sentiment (praise − complaints, % of rated mentions)",
                     zeroline=False)
    fig.update_yaxes(range=[0, top * 1.08], title_text="Reviews mentioning the theme")
    return _layout(fig, height=420)


def theme_radar(table: pd.DataFrame, name_a: str, name_b: str) -> go.Figure:
    """Radar of 0-100 theme satisfaction for two products (themes with enough mentions for both)."""
    data = table.dropna(subset=["satisfaction_a", "satisfaction_b"], how="all").head(8)
    themes = [t.replace(" & ", " &<br>") for t in data["theme"]]  # two-line labels fit narrow columns
    fig = go.Figure()
    for side, name, color in (("a", name_a, COMPARE_COLORS[0]), ("b", name_b, COMPARE_COLORS[1])):
        values = data[f"satisfaction_{side}"].fillna(0).tolist()
        fig.add_scatterpolar(
            r=values + values[:1], theta=themes + themes[:1], name=name, fill="toself", opacity=0.75,
            line=dict(color=color, width=2), marker=dict(size=7),
            hovertemplate=f"{name}<br>%{{theta}}: %{{r:.0f}}/100<extra></extra>",
        )
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], angle=90,
                                                 gridcolor="rgba(128,128,128,0.25)", tickfont=dict(size=10)),
                                 angularaxis=dict(gridcolor="rgba(128,128,128,0.25)", tickfont=dict(size=11))))
    fig = _layout(fig, height=460, legend=True)
    fig.update_layout(margin=dict(l=92, r=92, t=40, b=20),
                      legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5))
    return fig


def compare_ratings(reviews_a: pd.DataFrame, name_a: str, reviews_b: pd.DataFrame, name_b: str) -> go.Figure:
    """Share of each product's reviews per star rating, side by side."""
    fig = go.Figure()
    for reviews, name, color in ((reviews_a, name_a, COMPARE_COLORS[0]), (reviews_b, name_b, COMPARE_COLORS[1])):
        counts = reviews["rating"].round().astype(int).value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        share = (counts / max(len(reviews), 1) * 100).round(1)
        fig.add_bar(x=[f"{s}★" for s in share.index], y=share.values, name=name,
                    marker=dict(color=color, cornerradius=4),
                    hovertemplate=f"{name}<br>%{{x}}: %{{y}}% of reviews<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title_text="% of reviews", ticksuffix="%")
    return _layout(fig, height=280, legend=True)


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

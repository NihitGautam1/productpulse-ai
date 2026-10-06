"""PDF report: a shareable summary of the current selection.

Built with fpdf2. Charts are drawn directly as shapes, so no browser or image
library is needed. Hindi text needs a font with Devanagari: Nirmala UI (Windows)
or Noto Sans / DejaVu (Linux) are used when available; otherwise text is limited
to Latin characters.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from emotions import emotion_counts
from insights import evidence_for_theme, fix_impact, headline, health_score
from sentiment import sentiment_counts
from themes import theme_summary
from wishlist import extract_requests, group_requests

logging.getLogger("fontTools").setLevel(logging.ERROR)  # silences harmless font-subsetting notices

# (regular, bold, collection index of regular, collection index of bold)
FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\Nirmala.ttc", r"C:\Windows\Fonts\Nirmala.ttc", 0, 1),
    ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", 0, 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, 0),
]
DEVANAGARI_FONT = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
BLUE, RED, GREEN, GREY, AMBER =(42, 120, 214), (227, 73, 72), (27, 175, 122), (163, 162, 156), (237, 161, 0)
LEVEL_RGB = {"Critical": (208, 59, 59), "High": (236, 131, 90), "Medium": (250, 178, 25), "Low": GREY}
EMOTION_RGB = {"Anger": (208, 59, 59), "Frustration": (235, 104, 52), "Disappointment": (164, 106, 184),
               "Confusion": AMBER, "Delight": GREEN, "Satisfaction": BLUE}
TEXT, MUTED = (26, 26, 25), (110, 110, 105)
NON_BMP = re.compile(r"[\U00010000-\U0010FFFF\u2600-\u27BF\uFE0F]")


class ReportPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(15, 15, 15)
        self.has_unicode = False
        for regular, bold, regular_index, bold_index in FONT_CANDIDATES:
            if Path(regular).exists() and Path(bold).exists():
                self.add_font("Body", fname=regular, collection_font_number=regular_index)
                self.add_font("Body", style="B", fname=bold, collection_font_number=bold_index)
                self.has_unicode = True
                break
        self.body_family = "Body" if self.has_unicode else "Helvetica"
        # Linux fonts (Noto Sans, DejaVu) have no Devanagari: borrow it from Noto Sans Devanagari for Hindi text.
        if self.has_unicode and Path(DEVANAGARI_FONT).exists():
            self.add_font("Deva", fname=DEVANAGARI_FONT)
            self.set_fallback_fonts(["Deva"], exact_match=False)

    def clean_text(self, text) -> str:
        """Remove emoji, and anything the core font cannot show when no Unicode font is available."""
        text = NON_BMP.sub("", str(text)).replace("★", "*").replace("☆", "")
        if not self.has_unicode:
            text = (text.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-")
                    .replace("…", "...").replace("·", "-"))
            text = text.encode("latin-1", "replace").decode("latin-1")
        return text

    def use_font(self, size: float, bold: bool = False, color=TEXT) -> None:
        self.set_font(self.body_family, "B" if bold else "", size)
        self.set_text_color(*color)

    def footer(self) -> None:
        self.set_y(-11)
        self.use_font(8, color=MUTED)
        self.cell(0, 5, self.clean_text(f"ProductPulse AI · page {self.page_no()} of {{nb}}"), align="C")

    def heading(self, title: str, caption: str = "") -> None:
        if self.get_y() > 250:
            self.add_page()
        self.ln(4)
        self.use_font(13, bold=True)
        self.cell(0, 7, self.clean_text(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if caption:
            self.use_font(8.5, color=MUTED)
            self.multi_cell(0, 4.2, self.clean_text(caption), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.5)

    def para(self, text: str, size: float = 10, color=TEXT, bold: bool = False, indent: float = 0) -> None:
        self.use_font(size, bold, color)
        self.set_x(self.l_margin + indent)
        self.multi_cell(self.epw - indent, size * 0.5, self.clean_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                        markdown=True)

    def add_bullet(self, text: str, color=BLUE, size: float = 9.5) -> None:
        if self.get_y() > self.page_break_trigger - 10:  # keep the dot on the same page as its text
            self.add_page()
        y = self.get_y()
        self.set_fill_color(*color)
        self.ellipse(self.l_margin + 1, y + 1.6, 1.8, 1.8, style="F")
        self.para(text, size=size, indent=5)
        self.ln(0.8)


def _bold_markers(text: str) -> str:
    """Keep **bold**, turn __problem__ markers into bold too (fpdf2 markdown)."""
    return re.sub(r"__(.+?)__", r"**\1**", str(text))


def _plain_markdown(text: str) -> list[tuple[str, str]]:
    """AI Markdown as (kind, text) lines: heading, bullet or text."""
    lines = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line or line.startswith("_("):
            continue
        if line.startswith("#"):
            lines.append(("heading", line.lstrip("# ").strip()))
        elif re.match(r"^([-*]|\d+\.)\s+", line):
            lines.append(("bullet", re.sub(r"^([-*]|\d+\.)\s+", "", line)))
        else:
            lines.append(("text", line))
    return lines


def build_report(ctx, ai_summary: str | None = None) -> bytes:
    """PDF bytes for the current selection. `ctx` is a views.Context."""
    df, mentions, priority = ctx.reviews, ctx.mentions, ctx.priority
    pdf = ReportPDF()
    pdf.set_title(f"ProductPulse AI report - {ctx.subject}")
    pdf.set_author("ProductPulse AI")
    pdf.add_page()

    # Header band.
    pdf.set_fill_color(*BLUE)
    pdf.rect(0, 0, pdf.w, 30, style="F")
    pdf.set_xy(15, 8)
    pdf.use_font(19, bold=True, color=(255, 255, 255))
    pdf.cell(0, 9, pdf.clean_text("ProductPulse AI · Review Report"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.use_font(10, color=(225, 236, 250))
    pdf.cell(0, 6, pdf.clean_text(f"{ctx.subject.title() if ctx.subject == 'all products' else ctx.subject} · {ctx.date_label}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(34)
    pdf.use_font(8.5, color=MUTED)
    note = f"Generated {pd.Timestamp.now():%d %b %Y, %H:%M} · {ctx.source_label} · {len(df):,} reviews"
    if ctx.is_sample:
        note += " · Demo data: synthetic reviews, not real customer feedback"
    pdf.multi_cell(0, 4.2, pdf.clean_text(note), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if df.empty:
        pdf.ln(6)
        pdf.para("No reviews match the current filters.")
        return bytes(pdf.output())

    # Health score and headline.
    score = health_score(df)
    color = GREEN if score >= 70 else AMBER if score >= 50 else RED
    top = pdf.get_y() + 4
    pdf.set_fill_color(*color)
    pdf.ellipse(15, top, 26, 26, style="F")
    pdf.set_fill_color(255, 255, 255)
    pdf.ellipse(18, top + 3, 20, 20, style="F")
    pdf.set_xy(15, top + 7.5)
    pdf.use_font(15, bold=True)
    pdf.cell(26, 7, str(score), align="C")
    pdf.set_xy(15, top + 14)
    pdf.use_font(6.5, color=MUTED)
    pdf.cell(26, 4, "HEALTH", align="C")
    pdf.set_xy(46, top + 1)
    pdf.use_font(12.5, bold=False)
    pdf.multi_cell(pdf.w - 61, 6, pdf.clean_text(_bold_markers(headline(df, mentions, priority))), markdown=True,
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(max(pdf.get_y(), top + 28))

    # Key numbers.
    counts = sentiment_counts(df)
    total = len(df)
    kpis = [("Reviews", f"{total:,}"), ("Average rating", f"{df['rating'].mean():.2f} / 5"),
            ("Positive", f"{counts['Positive'] / total:.0%}"), ("Negative", f"{counts['Negative'] / total:.0%}"),
            ("Health score", f"{score}/100")]
    width = (pdf.epw - 4 * 3) / 5
    y = pdf.get_y() + 2
    for i, (label, value) in enumerate(kpis):
        x = 15 + i * (width + 3)
        pdf.set_draw_color(220, 220, 215)
        pdf.set_fill_color(247, 247, 244)
        pdf.rect(x, y, width, 17, style="DF", round_corners=True, corner_radius=2)
        pdf.set_xy(x + 3, y + 2.5)
        pdf.use_font(7.5, color=MUTED)
        pdf.cell(width - 6, 4, pdf.clean_text(label))
        pdf.set_xy(x + 3, y + 7.5)
        pdf.use_font(13, bold=True)
        pdf.cell(width - 6, 7, pdf.clean_text(value))
    pdf.set_y(y + 20)

    # Sentiment and emotions.
    pdf.heading("How customers feel", "Sentiment of every review, and the main emotion detected in each one.")
    bar_y = pdf.get_y()
    x = 15
    for label, rgb in (("Positive", BLUE), ("Neutral", GREY), ("Negative", RED)):
        share = counts[label] / total
        pdf.set_fill_color(*rgb)
        if share > 0:
            pdf.rect(x, bar_y, pdf.epw * share, 7, style="F")
            if share >= 0.08:
                pdf.set_xy(x, bar_y + 1.2)
                pdf.use_font(8, bold=True, color=(255, 255, 255))
                pdf.cell(pdf.epw * share, 4.5, f"{label} {share:.0%}", align="C")
        x += pdf.epw * share
    pdf.set_y(bar_y + 10)
    emotion_totals = emotion_counts(df).drop("No clear emotion", errors="ignore")
    emotion_totals = emotion_totals[emotion_totals > 0].sort_values(ascending=False)
    if len(emotion_totals):
        biggest = emotion_totals.max()
        for name, count in emotion_totals.items():
            y = pdf.get_y()
            pdf.set_xy(15, y)
            pdf.use_font(9)
            pdf.cell(32, 5, pdf.clean_text(name))
            pdf.set_fill_color(*EMOTION_RGB.get(name, GREY))
            pdf.rect(48, y + 0.8, max(1, 100 * count / biggest), 3.6, style="F")
            pdf.set_xy(50 + 100 * count / biggest, y)
            pdf.use_font(8, color=MUTED)
            pdf.cell(30, 5, f"{count} ({count / total:.0%})")
            pdf.set_y(y + 5.6)

    # Priority issues.
    pdf.heading("Priority issues", "Ranked by the priority score (0-100). 'If fixed' = how much the average rating "
                                   "could rise if reviews complaining about it rated like the rest (an optimistic estimate).")
    if priority.empty:
        pdf.para("No complaints detected in this selection.")
    else:
        impact = fix_impact(df, mentions, priority).set_index("theme")["gain"]
        headers = [("Issue", 48), ("Priority", 22), ("Reviews", 18), ("Avg rating", 20), ("Score", 50), ("If fixed", 22)]
        pdf.use_font(8, bold=True, color=MUTED)
        for title, w in headers:
            pdf.cell(w, 6, title)
        pdf.ln(6)
        for row in priority.head(8).itertuples():
            if pdf.get_y() > 270:
                pdf.add_page()
            y = pdf.get_y()
            pdf.set_draw_color(230, 230, 226)
            pdf.line(15, y, 15 + pdf.epw, y)
            pdf.set_xy(15, y + 1)
            pdf.use_font(9, bold=True)
            pdf.cell(48, 6, pdf.clean_text(row.theme))
            pdf.set_fill_color(*LEVEL_RGB.get(row.level, GREY))
            pdf.rect(63, y + 2, 18, 4.6, style="F", round_corners=True, corner_radius=2)
            pdf.set_xy(63, y + 2)
            pdf.use_font(7.5, bold=True, color=(26, 26, 25) if row.level == "Medium" else (255, 255, 255))
            pdf.cell(18, 4.6, row.level, align="C")
            pdf.set_xy(85, y + 1)
            pdf.use_font(9)
            pdf.cell(18, 6, str(row.complaint_reviews))
            pdf.cell(20, 6, f"{row.avg_rating:.2f}")
            pdf.set_fill_color(235, 235, 231)
            pdf.rect(123, y + 2.6, 40, 3, style="F")
            pdf.set_fill_color(*LEVEL_RGB.get(row.level, GREY))
            pdf.rect(123, y + 2.6, 40 * min(row.score, 100) / 100, 3, style="F")
            pdf.set_xy(165, y + 1)
            pdf.use_font(8, color=MUTED)
            pdf.cell(8, 6, f"{row.score:.0f}")
            pdf.set_xy(173, y + 1)
            pdf.use_font(9, bold=True, color=GREEN)
            gain = impact.get(row.theme)
            pdf.cell(22, 6, f"+{gain:.2f}" if gain is not None and gain > 0 else "-")
            pdf.set_y(y + 8)
        top_issue = priority.iloc[0]["theme"]
        quotes = evidence_for_theme(mentions, top_issue, "complaint", 2)
        if not quotes.empty:
            pdf.ln(2)
            pdf.para(f"What customers say about {top_issue}:", size=9.5, bold=True)
            for _, quote in quotes.iterrows():
                text = str(quote["review_text"])
                text = text if len(text) <= 300 else text[:300].rsplit(" ", 1)[0] + " ..."
                pdf.para(f"“{text}” ({quote['review_id']}, {int(quote['rating'])}/5)", size=9, color=MUTED, indent=4)
                pdf.ln(1)

    # Praise.
    summary = theme_summary(mentions, total)
    praised = summary[summary["praise"] >= 2].sort_values("praise", ascending=False).head(4)
    if not praised.empty:
        pdf.heading("What customers love")
        for row in praised.itertuples():
            pdf.add_bullet(f"**{row.theme}**: praised in {row.praise} reviews ({row.praise_pct}%)", color=BLUE)

    # Key insights.
    insights = [i for i in ctx.insights if i["kind"] != "overview"][:6]
    if insights:
        pdf.heading("Key insights", "Computed from the data; review IDs are the evidence.")
        kind_rgb = {"praise": BLUE, "complaint": RED, "trend": (235, 104, 52), "isolated": GREY, "language": GREEN}
        for item in insights:
            evidence = f" (evidence: {', '.join(item['evidence'])})" if item["evidence"] else ""
            pdf.add_bullet(item["text"] + evidence, color=kind_rgb.get(item["kind"], GREY))

    # Feature wishlist.
    wishes = group_requests(extract_requests(df)).head(5)
    if not wishes.empty:
        pdf.heading("Feature wishlist", "What customers ask to be added or changed, most requested first.")
        for row in wishes.itertuples():
            pdf.add_bullet(f"**{row.request}** ({row.reviews} review{'s' if row.reviews != 1 else ''})", color=GREEN)

    # AI summary.
    if ai_summary:
        pdf.heading("AI summary", "Generated by AI from the reviews. Verify important points against the evidence.")
        for kind, text in _plain_markdown(ai_summary):
            if kind == "heading":
                pdf.ln(1)
                pdf.para(text, size=10.5, bold=True)
            elif kind == "bullet":
                pdf.add_bullet(text, color=BLUE)
            else:
                pdf.para(text, size=9.5)
    return bytes(pdf.output())

"""Upload interface: several review CSV files at once, with automatic column detection,
manual mapping when needed, duplicate handling and an upload summary.

The CSV logic itself lives in csv_normalizer.py; this module only handles the Streamlit side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

import csv_normalizer as cn
from data_loader import REQUIRED_COLUMNS

KEEP_ALL = "Keep all"
REMOVE_EXACT = "Remove exact duplicates"
UPLOAD_HELP = ("Upload one or more CSV files. ProductPulse AI will automatically detect and normalize common "
               "column names.")
FORMATS_NOTE = """**Supported column formats**

Standard: `review_id | product_name | rating | review_text | review_date`

Alternative: `id | product | stars | review | date`

Other common names (such as *Review ID*, *score*, *comment*, *feedback*, *created_at*) are recognised too."""


@dataclass
class UploadState:
    results: list = field(default_factory=list)           # csv_normalizer.FileResult per file
    combined: cn.Combined | None = None

    @property
    def usable(self) -> bool:
        return self.combined is not None and self.combined.reviews is not None

    @property
    def signature(self) -> str:
        return "|".join(r.key for r in self.results)

    @property
    def needs_attention(self) -> bool:
        return any(r.status != "ok" for r in self.results) or bool(
            self.combined and (self.combined.exact_duplicates or self.combined.conflicting_ids)) or any(
            r.scale not in ("1-5", "0-5", "unknown") for r in self.results if r.status == "ok")


@st.cache_data(show_spinner="Reading CSV files...", max_entries=64)
def _process(data: bytes, name: str, mapping: tuple, convert_scale: bool) -> cn.FileResult:
    return cn.process_file(data, name, dict(mapping) if mapping else None, convert_scale)


def uploader() -> list:
    """The sidebar file picker. Returns the uploaded files (possibly empty)."""
    files = st.file_uploader("Upload Review CSV Files", type="csv", accept_multiple_files=True, key="csv_files",
                             help=UPLOAD_HELP)
    st.caption(UPLOAD_HELP)
    with st.expander("Supported column formats"):
        st.markdown(FORMATS_NOTE)
    return files or []


def process_uploads(files: list) -> UploadState:
    """Normalise every file (using any manual mappings and choices made earlier) and combine them."""
    mappings = st.session_state.setdefault("csv_mappings", {})
    scale_choices = st.session_state.setdefault("csv_convert_scale", {})
    state = UploadState()
    for upload in files:
        data = upload.getvalue()
        key = cn.file_key(upload.name, data)
        mapping = tuple(sorted(mappings[key].items())) if key in mappings else ()
        state.results.append(_process(data, upload.name, mapping, scale_choices.get(key, True)))
    remove = st.session_state.get("csv_duplicates", KEEP_ALL) == REMOVE_EXACT
    state.combined = cn.combine_uploaded_files(state.results, remove_exact_duplicates=remove)
    return state


def sidebar_status(state: UploadState) -> None:
    ok = [r for r in state.results if r.status == "ok"]
    problems = len(state.results) - len(ok)
    if state.usable:
        st.success(f"{len(ok)} of {len(state.results)} file(s) · {len(state.combined.reviews):,} reviews", icon="📂")
    if problems:
        st.warning(f"{problems} file(s) need attention: see the upload panel on the page.", icon="⚠️")


# ---------------------------------------------------------------- main-area panel

def render_panel(state: UploadState) -> None:
    """Upload summary, plus mapping forms and choices for anything that needs the user."""
    if not state.results:
        return
    title = f"📂 Upload summary · {len(state.results)} file(s)"
    if state.usable:
        title += f" · {len(state.combined.reviews):,} reviews"
    # Open the first time a set of files is seen, and whenever a file still has a problem.
    first_view = st.session_state.get("upload_panel_seen") != state.signature
    problem = any(r.status != "ok" for r in state.results)
    with st.expander(title, expanded=first_view or problem):
        st.session_state["upload_panel_seen"] = state.signature
        _summary(state)
        _choices(state)
    for result in state.results:
        if result.status == "needs_mapping":
            mapping_form(result)


def _summary(state: UploadState) -> None:
    left, right = st.columns((1.4, 1), gap="large")
    with left:
        st.markdown(f"**Uploaded files: {len(state.results)}**")
        for result in state.results:
            if result.status == "ok":
                st.markdown(f"✅ **{result.name}**: {len(result.dataframe):,} reviews "
                            f"(from {result.rows_in:,} rows)")
                _mapping_caption(result)
            elif result.status == "needs_mapping":
                st.markdown(f"🧩 **{result.name}**: needs column mapping. {result.reason}")
            else:
                st.markdown(f"❌ **Could not process: {result.name}**  \nReason: {result.reason}")
    with right:
        if state.usable:
            reviews = state.combined.reviews
            c1, c2 = st.columns(2)
            c1.metric("Total reviews", f"{len(reviews):,}")
            c2.metric("Products detected", reviews["product_name"].nunique())
            st.markdown("**Columns normalized**  \n" + "  \n".join(f"✓ {c}" for c in REQUIRED_COLUMNS))
    warnings = []
    combined = state.combined
    if combined and combined.duplicate_ids:
        warnings.append(f"{combined.duplicate_ids} duplicate review ID(s)")
    for result in state.results:
        if result.status != "ok":
            continue
        if result.missing_dates:
            warnings.append(f"{result.name}: {result.missing_dates} review(s) missing dates (left out)")
        warnings += [f"{result.name}: {w}" for w in result.warnings
                     if "invalid date" not in w]  # already covered by the missing-dates line
        warnings += [f"{result.name}: {n}" for n in result.detection.notes]
    if combined:
        warnings += combined.warnings
    if warnings:
        st.markdown("**Warnings**")
        for warning in warnings:
            st.caption("⚠️ " + warning)


def _mapping_caption(result: cn.FileResult) -> None:
    detection = result.detection
    parts = []
    for field_name in REQUIRED_COLUMNS:
        source = detection.mapping[field_name]
        shown = dict(cn.NO_COLUMN_CHOICES.values()).get(source, f"“{source}”")
        method = detection.methods.get(field_name, "")
        flag = " ⚑" if method in ("content", "similar") else ""
        parts.append(f"{cn.FIELD_LABELS[field_name]} ← {shown}{flag}")
    st.caption(" · ".join(parts) + ("  \n⚑ = detected from the values or a similar name; check it is right."
                                    if any(m in ("content", "similar") for m in detection.methods.values()) else ""))
    if st.toggle("Change mapping", key=f"edit_{result.key}"):
        mapping_form(result, editing=True)


def _choices(state: UploadState) -> None:
    combined = state.combined
    if combined and (combined.exact_duplicates or combined.duplicate_ids):
        st.markdown("**Duplicate review IDs**")
        st.caption(f"{combined.duplicate_ids} review ID(s) appear more than once; {combined.exact_duplicates} "
                   "row(s) are exact copies (every field identical). Reviews that share an ID but differ are "
                   "always kept.")
        st.radio("Duplicates", [KEEP_ALL, REMOVE_EXACT], key="csv_duplicates", horizontal=True,
                 label_visibility="collapsed")
    scale_choices = st.session_state.setdefault("csv_convert_scale", {})
    for result in state.results:
        if result.status == "ok" and result.scale not in ("1-5", "0-5", "unknown"):
            value = st.checkbox(f"Convert **{result.name}** ratings from a {result.scale} scale to 1-5 stars",
                                value=scale_choices.get(result.key, True), key=f"scale_{result.key}")
            if value != scale_choices.get(result.key, True):
                scale_choices[result.key] = value
                st.rerun()


def mapping_form(result: cn.FileResult, editing: bool = False) -> None:
    """Dropdowns to choose which CSV column holds each field."""
    columns = list(result.raw.columns)
    detection = result.detection
    with st.container(border=True):
        if not editing:
            st.markdown(f"#### 🧩 Column mapping needed: {result.name}")
            st.markdown("ProductPulse could not automatically identify: "
                        + ", ".join(f"**{cn.FIELD_LABELS[f]}**" for f in detection.missing))
        st.caption("A preview of the file:")
        st.dataframe(result.raw.head(5), hide_index=True, width="stretch")
        with st.form(f"mapping_{result.key}_{editing}"):
            choices = {}
            cols = st.columns(len(REQUIRED_COLUMNS))
            for col, field_name in zip(cols, REQUIRED_COLUMNS):
                options = cn.column_options(field_name, columns)
                values = [v for v, _ in options]
                labels = dict(options)
                current = detection.mapping.get(field_name)
                choices[field_name] = col.selectbox(
                    cn.FIELD_LABELS[field_name], values, index=values.index(current) if current in values else None,
                    format_func=lambda v, labels=labels: labels[v], placeholder="Select CSV column",
                    key=f"map_{result.key}_{field_name}_{editing}")
            submitted = st.form_submit_button("Apply Mapping", type="primary")
        if submitted:
            problems = cn.check_manual_mapping(choices, columns)
            if problems:
                for problem in problems:
                    st.error(problem)
            else:
                st.session_state.setdefault("csv_mappings", {})[result.key] = choices
                st.rerun()

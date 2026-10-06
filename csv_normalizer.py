"""Turn review CSV files from different sources into the standard ProductPulse schema.

    input CSV  ->  column detection / mapping  ->  review_id, product_name, rating,
                                                   review_text, review_date  ->  existing pipeline

Detection works in steps, from most to least certain:
1. Known column names (aliases), ignoring case, spaces, underscores and hyphens.
2. Close spellings of those names ("Review Rating", "reviewText", "Stars (1-5)").
3. The values in the column (ratings look like small numbers, dates parse as dates,
   review text is long free text) - only when exactly one column clearly fits.
Every name match is also checked against the values, so a "score" column full of text
is not used as the rating. When a required column cannot be identified confidently,
the file is marked as needing manual mapping instead of guessing.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import REQUIRED_COLUMNS, validate_reviews

FIELD_LABELS = {
    "review_id": "Review ID",
    "product_name": "Product Name",
    "rating": "Rating",
    "review_text": "Review Text",
    "review_date": "Review Date",
}

# Aliases per field, most specific first. Compared after normalising (lower case, letters and digits only).
ALIASES = {
    "review_id": ["review_id", "reviewid", "review id", "id", "review_number", "review_no", "review_number_id",
                  "review_num", "comment_id", "feedback_id", "reviews.id", "uid", "review"],
    "product_name": ["product_name", "product", "product name", "product_title", "product title", "item",
                     "item_name", "item name", "item_title", "title_of_product", "productname", "model", "name"],
    "rating": ["rating", "ratings", "stars", "star_rating", "star rating", "score", "review_rating",
               "review score", "review_score", "rating_value", "overall", "review_stars", "reviews.rating",
               "rate", "star", "user_rating", "customer_rating"],
    "review_text": ["review_text", "review", "review_texts", "review text", "comment", "comments", "feedback",
                    "feedback_text", "feedback text", "text", "content", "description", "review_body",
                    "review body", "body", "reviewtext", "review_content", "reviews.text", "message",
                    "customer_review", "customer_feedback", "opinion", "remarks"],
    "review_date": ["review_date", "review date", "date", "review_time", "timestamp", "created_at",
                    "created_date", "posted_date", "published_date", "reviews.date", "date_posted", "posted_at",
                    "submitted_at", "review_datetime", "unixreviewtime", "reviewtime", "time", "datetime",
                    "published_at", "date_added", "created"],
}
# Fall back to an identifier as the product only when there is no name-like product column at all.
PRODUCT_ID_ALIASES = ["product_id", "productid", "asin", "sku", "item_id", "itemid"]
# Words that suggest a field when the full name is not a known alias (e.g. "Review Rating (out of 5)").
NAME_HINTS = {
    "rating": ["rating", "stars", "star", "score"],
    "review_date": ["date", "time", "posted", "created", "published"],
    "review_text": ["review", "comment", "feedback", "text", "body", "content"],
    "product_name": ["product", "item"],
    "review_id": ["id"],
}

# Choices for a field that has no column in the file (offered in manual mapping).
AUTO_ID = "__auto_id__"
FILE_AS_PRODUCT = "__file_name__"
RATING_FROM_TEXT = "__estimate_rating__"
DATE_TODAY = "__today__"
NO_COLUMN_CHOICES = {
    "review_id": (AUTO_ID, "(none - create IDs automatically)"),
    "product_name": (FILE_AS_PRODUCT, "(none - use the file name as the product)"),
    "rating": (RATING_FROM_TEXT, "(none - estimate stars from the review text)"),
    "review_date": (DATE_TODAY, "(none - use today's date)"),
}
SPECIAL_VALUES = {value for value, _ in NO_COLUMN_CHOICES.values()}
METHOD_LABELS = {"name": "column name", "similar": "similar column name", "content": "column values",
                 "auto": "created automatically", "manual": "your mapping", "default": "your choice"}

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
STAR_SYMBOLS = re.compile(r"[★⭐]")
FRACTION = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:/|out\s+of|of)\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# What a rating value looks like: "5", "4.5", "5 stars", "4/5", "8 out of 10", "★★★★".
RATING_SHAPE = re.compile(r"^\s*(?:\d+(?:[.,]\d+)?\s*(?:stars?|/\s*\d+(?:[.,]\d+)?|out\s+of\s+\d+(?:[.,]\d+)?)?"
                          r"|[★⭐☆\s]+)\s*$", re.IGNORECASE)
DATE_SHAPE = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|[A-Za-z]{3,9}\.? \d{1,2},? \d{2,4}|\d{1,2} [A-Za-z]{3,9}\.? \d{2,4}|\d{4}-\d{2}")
SCALES = {
    "1-5": None,
    "0-5": None,  # a 0 is simply an invalid rating; nothing to convert
    "1-10": lambda r: 1 + (r - 1) * 4 / 9,
    "0-10": lambda r: 1 + r * 4 / 10,
    "0-100": lambda r: 1 + r * 4 / 100,
}


class CSVReadError(Exception):
    """A file that cannot be read as a CSV at all (empty, binary, broken)."""


def normalize_name(name) -> str:
    """'Review ID', 'review_id', 'ReviewID' and 'review-id' all become 'reviewid'."""
    return re.sub(r"[^0-9a-z]", "", str(name).lower())


_ALIAS_INDEX: dict[str, list[tuple[str, int]]] = {}
for _field, _names in ALIASES.items():
    for _rank, _name in enumerate(_names):
        _ALIAS_INDEX.setdefault(normalize_name(_name), []).append((_field, _rank))


# ---------------------------------------------------------------- reading

def read_csv_bytes(data: bytes, name: str = "file.csv") -> pd.DataFrame:
    """Read CSV bytes, trying common encodings and separators (comma, semicolon, tab).

    Every column is read as text so values such as IDs keep their exact form.
    Raises CSVReadError with a friendly reason.
    """
    if not data or not data.strip():
        raise CSVReadError("The file is empty.")
    last_error = None
    for encoding in ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        try:
            frame = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str, skipinitialspace=True,
                                keep_default_na=False, na_values=[""])
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, TypeError, StopIteration) as exc:
            last_error = exc
            # The separator sniffer can fail on single-column or unusual files: try a plain comma read.
            try:
                frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, na_values=[""])
            except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc2:
                last_error = exc2
                continue
        frame.columns = [str(c).strip() for c in frame.columns]
        frame = frame.loc[:, [not str(c).startswith("Unnamed:") or frame[c].notna().any() for c in frame.columns]]
        if frame.empty or not len(frame.columns):
            raise CSVReadError("The file has column names but no rows." if len(frame.columns) else "The file is empty.")
        return frame
    raise CSVReadError(f"Could not read the file as CSV ({type(last_error).__name__}: {last_error}).")


# ---------------------------------------------------------------- value parsing

def parse_rating_value(value) -> float:
    """5, "5", "5 stars", "4/5", "8 out of 10", "4.5", "★★★★" -> a number (fractions become 0-5)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    if STAR_SYMBOLS.search(text) and not NUMBER.search(text):
        return float(len(STAR_SYMBOLS.findall(text)))
    fraction = FRACTION.search(text)
    if fraction:
        top, bottom = (float(x.replace(",", ".")) for x in fraction.groups())
        return round(top / bottom * 5, 2) if bottom > 0 else np.nan
    number = NUMBER.search(text)
    return float(number.group().replace(",", ".")) if number else np.nan


def parse_ratings(series: pd.Series) -> pd.Series:
    return series.map(parse_rating_value).astype(float)


def detect_scale(ratings: pd.Series) -> str:
    """Rating scale of parsed ratings: '1-5', '0-5', '1-10', '0-10', '0-100' or 'unknown'."""
    values = ratings.dropna()
    if values.empty:
        return "unknown"
    low, high = float(values.min()), float(values.max())
    if high <= 5:
        return "0-5" if low == 0 else "1-5"
    if high <= 10:
        return "0-10" if low < 1 else "1-10"
    if high <= 100:
        return "0-100"
    return "unknown"


def convert_scale(ratings: pd.Series, scale: str) -> pd.Series:
    """Convert ratings on `scale` to 1-5 (rounded to one decimal)."""
    convert = SCALES.get(scale)
    return ratings if convert is None else convert(ratings).round(1)


def parse_dates(series: pd.Series) -> pd.Series:
    """Dates in common formats, ISO timestamps (with time zones) or Unix seconds/milliseconds; NaT if unparseable."""
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    seconds = numeric.between(1e9, 2.2e9).fillna(False).astype(bool)
    millis = numeric.between(1e12, 2.2e12).fillna(False).astype(bool)
    if seconds.any():
        result[seconds] = pd.to_datetime(numeric[seconds], unit="s")
    if millis.any():
        result[millis] = pd.to_datetime(numeric[millis], unit="ms")
    rest = (~(seconds | millis) & text.notna() & (text != "")).fillna(False).astype(bool)
    if rest.any():
        parsed = pd.to_datetime(text[rest], errors="coerce", format="mixed", utc=True)
        result[rest] = parsed.dt.tz_convert(None)
    return result


# ---------------------------------------------------------------- column profiles

def _non_empty(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    return values[values != ""]


def looks_like_rating(series: pd.Series, min_share: float = 0.9) -> bool:
    values = _non_empty(series)
    if values.empty:
        return False
    shaped = values.str.match(RATING_SHAPE)
    parsed = parse_ratings(values[shaped])
    valid = parsed.dropna()
    # Values must look like ratings, not merely contain a number ("iPhone 15", "u1", "Samsung S24").
    if shaped.mean() < min_share or valid.empty or valid.min() < 0 or valid.max() > 100:
        return False
    return valid.nunique() <= 21  # ratings repeat a few values


def looks_like_date(series: pd.Series, min_share: float = 0.8) -> bool:
    values = _non_empty(series)
    if values.empty:
        return False
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= 0.9:  # numbers only count as dates if they are Unix timestamps
        return (numeric.between(1e9, 2.2e9) | numeric.between(1e12, 2.2e12)).mean() >= 0.9
    shaped = values.str.contains(DATE_SHAPE).mean() >= min_share
    return shaped and parse_dates(values).notna().mean() >= min_share and values.str.len().mean() <= 40


def looks_like_text(series: pd.Series) -> bool:
    values = _non_empty(series)
    if values.empty:
        return False
    return values.str.len().mean() >= 20 and values.str.split().str.len().mean() >= 4 and not looks_like_date(values)


def looks_like_id(series: pd.Series) -> bool:
    values = _non_empty(series)
    if values.empty:
        return False
    return (values.nunique() / len(values) >= 0.95 and values.str.len().mean() <= 40
            and values.str.contains(r"\s").mean() <= 0.1)


def looks_like_product(series: pd.Series) -> bool:
    values = _non_empty(series)
    if values.empty:
        return False
    return values.str.len().mean() <= 120 and not looks_like_date(values) and not looks_like_rating(values)


CONTENT_CHECKS = {
    "review_id": looks_like_id,
    "product_name": looks_like_product,
    "rating": looks_like_rating,
    "review_text": looks_like_text,
    "review_date": looks_like_date,
}


def _id_accepts(series: pd.Series) -> bool:
    """A name-matched ID column only needs to be short values; duplicates are handled later."""
    values = _non_empty(series)
    return values.empty or values.str.len().mean() <= 60


# ---------------------------------------------------------------- detection

@dataclass
class Detection:
    """Which CSV column (or automatic choice) feeds each standard field, and how it was found."""
    mapping: dict[str, str] = field(default_factory=dict)   # field -> column name or a special value
    methods: dict[str, str] = field(default_factory=dict)   # field -> name | similar | content | auto
    missing: list[str] = field(default_factory=list)        # required fields that need a manual choice
    notes: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


NAME_MATCH_SHARE = 0.5  # a column matched by name only needs most of its values to fit (bad ones are cleaned out)


def _accepts(field_name: str, series: pd.Series) -> bool:
    """Value check for a column already matched by its name (more lenient than detection by values alone)."""
    if field_name == "review_id":
        return _id_accepts(series)
    if field_name == "rating":
        return looks_like_rating(series, NAME_MATCH_SHARE)
    if field_name == "review_date":
        return looks_like_date(series, NAME_MATCH_SHARE)
    return CONTENT_CHECKS[field_name](series)


def detect_column_mapping(df: pd.DataFrame) -> Detection:
    """Work out which column holds each standard field. Never guesses when unsure: unsure fields go in `missing`."""
    detection = Detection()
    columns = list(df.columns)
    used: set[str] = set()

    def assign(field_name: str, column: str, method: str) -> None:
        detection.mapping[field_name] = column
        detection.methods[field_name] = method
        used.add(column)

    # 1. Exact alias matches. Each field takes its best-ranked candidate whose values fit.
    candidates: dict[str, list[tuple[int, str]]] = {}
    for column in columns:
        for field_name, rank in _ALIAS_INDEX.get(normalize_name(column), []):
            candidates.setdefault(field_name, []).append((rank, column))
    # Fields with an unambiguous alias go first, so an ambiguous name such as "review" or "name"
    # is left for the field its values actually fit.
    order = sorted(candidates, key=lambda f: (min(r for r, _ in candidates[f]) != 0, REQUIRED_COLUMNS.index(f)))
    for field_name in order:
        for rank, column in sorted(candidates[field_name]):
            if column not in used and _accepts(field_name, df[column]):
                assign(field_name, column, "name")
                break
            if column not in used and rank == 0:
                detection.notes.append(f"Column “{column}” is named like {FIELD_LABELS[field_name]} but its "
                                       "values do not look like it, so it was not used automatically.")

    # 2. Similar names: close spellings of an alias, or a hint word inside the name, checked against the values.
    for field_name in REQUIRED_COLUMNS:
        if field_name in detection.mapping:
            continue
        alias_names = [normalize_name(a) for a in ALIASES[field_name]]
        for column in columns:
            if column in used:
                continue
            norm = normalize_name(column)
            close = difflib.get_close_matches(norm, alias_names, n=1, cutoff=0.85)
            hinted = any(hint in norm for hint in NAME_HINTS[field_name]) and (
                field_name != "review_id" or norm.endswith("id"))
            if (close or hinted) and _accepts(field_name, df[column]):
                assign(field_name, column, "similar")
                break

    # 3. Values only, and only when exactly one unused column fits (never for product or ID: too risky).
    for field_name in ("review_text", "rating", "review_date"):
        if field_name in detection.mapping:
            continue
        fits = [c for c in columns if c not in used and CONTENT_CHECKS[field_name](df[c])]
        if field_name == "review_text" and len(fits) > 1:
            # Several text columns (e.g. a title and a body): take the clearly longest one, otherwise ask.
            lengths = sorted(((_non_empty(df[c]).str.len().mean(), c) for c in fits), reverse=True)
            fits = [lengths[0][1]] if lengths[0][0] >= 1.5 * lengths[1][0] else fits
        if len(fits) == 1:
            assign(field_name, fits[0], "content")

    # 4. A product identifier stands in for the product name only if nothing else was found.
    if "product_name" not in detection.mapping:
        for column in columns:
            if column not in used and normalize_name(column) in {normalize_name(a) for a in PRODUCT_ID_ALIASES}:
                assign("product_name", column, "similar")
                detection.notes.append(f"No product name column, so the product identifier “{column}” is used "
                                       "as the product name.")
                break

    # 5. Review IDs can always be created safely.
    if "review_id" not in detection.mapping:
        detection.mapping["review_id"] = AUTO_ID
        detection.methods["review_id"] = "auto"
        detection.notes.append("No review ID column was found, so IDs were created from the file name.")

    detection.missing = [f for f in REQUIRED_COLUMNS if f not in detection.mapping]
    return detection


def manual_detection(mapping: dict[str, str]) -> Detection:
    """A Detection built from the user's manual choices."""
    detection = Detection(mapping=dict(mapping))
    detection.methods = {f: ("default" if v in SPECIAL_VALUES else "manual") for f, v in mapping.items()}
    detection.missing = [f for f in REQUIRED_COLUMNS if not mapping.get(f)]
    return detection


def check_manual_mapping(mapping: dict[str, str], columns: list[str]) -> list[str]:
    """Problems with a manual mapping (empty list = fine)."""
    problems = []
    if not mapping.get("review_text") or mapping["review_text"] in SPECIAL_VALUES:
        problems.append("Choose the column that contains the review text.")
    chosen = [v for v in mapping.values() if v and v not in SPECIAL_VALUES]
    repeated = sorted({c for c in chosen if chosen.count(c) > 1})
    if repeated:
        problems.append("Each column can only be used once: " + ", ".join(f"“{c}”" for c in repeated) + ".")
    unknown = [c for c in chosen if c not in columns]
    if unknown:
        problems.append("Unknown column(s): " + ", ".join(unknown))
    for f in REQUIRED_COLUMNS:
        if not mapping.get(f):
            problems.append(f"Choose an option for {FIELD_LABELS[f]}.")
    return problems


# ---------------------------------------------------------------- normalising one file

@dataclass
class FileResult:
    """Outcome of processing one uploaded file."""
    name: str
    key: str                                    # stable per file content (name + hash)
    status: str = "ok"                          # ok | needs_mapping | error
    reason: str = ""
    raw: pd.DataFrame | None = None
    detection: Detection | None = None
    dataframe: pd.DataFrame | None = None       # normalised reviews (standard schema + source_file)
    warnings: list[str] = field(default_factory=list)
    scale: str = "1-5"
    missing_dates: int = 0
    rows_in: int = 0


def file_key(name: str, data: bytes) -> str:
    return f"{name}:{hashlib.sha1(data).hexdigest()[:12]}"


def normalize_columns(raw: pd.DataFrame, detection: Detection, file_name: str) -> pd.DataFrame:
    """Build the five standard columns (still as raw values) from a detection."""
    stem = Path(file_name).stem
    out = pd.DataFrame(index=raw.index)
    for field_name in REQUIRED_COLUMNS:
        source = detection.mapping[field_name]
        if source == AUTO_ID:
            out[field_name] = [f"{stem}-{i + 1}" for i in range(len(raw))]
        elif source == FILE_AS_PRODUCT:
            out[field_name] = stem
        elif source == DATE_TODAY:
            out[field_name] = pd.Timestamp.today().normalize()
        elif source == RATING_FROM_TEXT:
            out[field_name] = np.nan  # filled in after the text is cleaned
        else:
            out[field_name] = raw[source]
    return out


def normalize_dataframe(raw: pd.DataFrame, detection: Detection, file_name: str,
                        convert_rating_scale: bool = True) -> tuple[pd.DataFrame, list[str], str, int]:
    """Standard-schema reviews for one file. Returns (reviews, warnings, detected scale, rows missing a date).

    Cleaning (shared with single-file loading through data_loader.validate_reviews): empty rows and rows
    without review text are removed, ratings must be 1-5, dates must parse, whitespace is trimmed.
    """
    stem = Path(file_name).stem
    warnings: list[str] = []
    raw = raw.replace(r"^\s*$", np.nan, regex=True).dropna(how="all").reset_index(drop=True)
    if raw.empty:
        raise ValueError("The file has no rows with data.")
    frame = normalize_columns(raw, detection, file_name)

    for column in ("review_id", "product_name", "review_text"):
        frame[column] = frame[column].astype("string").str.strip()
    missing_ids = frame["review_id"].isna() | (frame["review_id"] == "")
    if missing_ids.any() and detection.mapping["review_id"] != AUTO_ID:
        frame.loc[missing_ids, "review_id"] = [f"{stem}-{i + 1}" for i in np.flatnonzero(missing_ids.to_numpy())]
        warnings.append(f"Created IDs for {int(missing_ids.sum())} review(s) without one.")

    if detection.mapping["rating"] == RATING_FROM_TEXT:
        from video_reviews import estimate_rating
        frame["rating"] = frame["review_text"].fillna("").map(estimate_rating).astype(float)
        warnings.append("No rating column: star ratings were estimated from the sentiment of each review.")
        scale = "1-5"
    else:
        frame["rating"] = parse_ratings(frame["rating"])
        scale = detect_scale(frame["rating"])
        if scale not in ("1-5", "0-5", "unknown"):
            if convert_rating_scale:
                frame["rating"] = convert_scale(frame["rating"], scale)
                warnings.append(f"Ratings were on a {scale} scale and were converted to 1-5.")
            else:
                warnings.append(f"Ratings are on a {scale} scale and were not converted, so ratings above 5 are "
                                "removed as invalid.")

    if detection.mapping["review_date"] != DATE_TODAY:
        frame["review_date"] = parse_dates(frame["review_date"])
    has_text = frame["review_text"].notna() & (frame["review_text"] != "")
    missing_dates = int(frame.loc[has_text, "review_date"].isna().sum())

    reviews, cleaning = validate_reviews(frame.assign(source_file=file_name))
    warnings.extend(cleaning)
    return reviews, warnings, scale, missing_dates


def process_file(data: bytes, name: str, mapping: dict[str, str] | None = None,
                 convert_rating_scale: bool = True) -> FileResult:
    """Read, detect and normalise one uploaded file. Never raises: problems are reported in the result."""
    result = FileResult(name=name, key=file_key(name, data))
    try:
        result.raw = read_csv_bytes(data, name)
    except CSVReadError as exc:
        result.status, result.reason = "error", str(exc)
        return result
    result.rows_in = len(result.raw)
    result.detection = manual_detection(mapping) if mapping else detect_column_mapping(result.raw)
    if not result.detection.complete:
        labels = ", ".join(FIELD_LABELS[f] for f in result.detection.missing)
        result.status, result.reason = "needs_mapping", f"Could not identify the {labels} column(s)."
        return result
    try:
        result.dataframe, result.warnings, result.scale, result.missing_dates = normalize_dataframe(
            result.raw, result.detection, name, convert_rating_scale)
    except (ValueError, KeyError) as exc:  # e.g. no valid rows remain after cleaning
        result.status, result.reason = "error", str(exc)
    return result


# ---------------------------------------------------------------- combining files

@dataclass
class Combined:
    reviews: pd.DataFrame | None
    duplicate_ids: int = 0          # review IDs that appear more than once
    exact_duplicates: int = 0       # rows identical to an earlier row
    conflicting_ids: int = 0        # IDs shared by reviews with different content (renamed)
    removed_duplicates: int = 0
    warnings: list[str] = field(default_factory=list)


def combine_uploaded_files(results: list[FileResult], remove_exact_duplicates: bool = False) -> Combined:
    """Combine the successfully normalised files.

    - Exact duplicates (every standard field identical) are kept, or removed if asked.
    - Different reviews that share an ID are both kept; later ones get a "~2", "~3" suffix so that
      evidence and AI citations always point to one specific review.
    """
    frames = [r.dataframe for r in results if r.status == "ok" and r.dataframe is not None]
    if not frames:
        return Combined(reviews=None)
    reviews = pd.concat(frames, ignore_index=True)
    combined = Combined(reviews=reviews)

    exact = reviews.duplicated(subset=REQUIRED_COLUMNS, keep="first")
    combined.exact_duplicates = int(exact.sum())
    combined.duplicate_ids = int(reviews["review_id"].duplicated(keep=False).groupby(reviews["review_id"]).any().sum())
    if remove_exact_duplicates and exact.any():
        reviews = reviews[~exact].reset_index(drop=True)
        combined.removed_duplicates = int(exact.sum())
        combined.warnings.append(f"Removed {int(exact.sum())} exact duplicate review(s).")

    # Same ID, different content: keep both, give the later ones a unique ID.
    distinct = reviews.drop_duplicates(subset=REQUIRED_COLUMNS)
    clashing_ids = distinct["review_id"][distinct["review_id"].duplicated()].unique()
    if len(clashing_ids):
        combined.conflicting_ids = len(clashing_ids)
        content_key = reviews[REQUIRED_COLUMNS].astype(str).agg("|".join, axis=1)
        for review_id in clashing_ids:
            rows = reviews.index[reviews["review_id"] == review_id]
            versions = list(dict.fromkeys(content_key[rows]))  # first-seen order
            for index in rows:
                version = versions.index(content_key[index])
                if version:
                    reviews.at[index, "review_id"] = f"{review_id}~{version + 1}"
        combined.warnings.append(
            f"{len(clashing_ids)} review ID(s) are used by different reviews (for example “{clashing_ids[0]}”). "
            "All of them were kept; later ones got a ~2, ~3 suffix so evidence links stay correct.")
    combined.reviews = reviews.sort_values("review_date", kind="stable").reset_index(drop=True)
    return combined


def column_options(field_name: str, columns: list[str]) -> list[tuple[str, str]]:
    """(value, label) choices for a manual-mapping dropdown."""
    options = [(c, c) for c in columns]
    if field_name in NO_COLUMN_CHOICES:
        options.append(NO_COLUMN_CHOICES[field_name])
    return options

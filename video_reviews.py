"""Video reviews: turn YouTube videos and uploaded video files into text reviews.

Each video becomes one review row whose text is the transcript, so every other
analysis (sentiment, themes, priority, trends, Q&A) works on it unchanged.

- YouTube links: the video's captions are used when it has them (free, no API key).
  Otherwise Gemini transcribes the video from its URL.
- Uploaded files: Gemini transcribes the speech (needs a Gemini API key).

Videos have no star rating, so unless the user enters one it is estimated from
the sentiment of the transcript and marked as estimated.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import ai_service
from sentiment import score_text

YOUTUBE_SOURCE = "YouTube video"
UPLOAD_SOURCE = "Uploaded video"
VIDEO_SOURCES = (YOUTUBE_SOURCE, UPLOAD_SOURCE)
VIDEO_COLUMNS = ["review_id", "product_name", "rating", "review_text", "review_date", "source",
                 "rating_estimated", "video_title", "video_url", "transcript_method"]

# Formats Gemini accepts for video and audio, keyed by file extension.
VIDEO_MIME_TYPES = {
    "mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm", "mkv": "video/x-matroska",
    "avi": "video/x-msvideo", "mpeg": "video/mpeg", "mpg": "video/mpg", "wmv": "video/wmv",
    "3gp": "video/3gpp", "flv": "video/x-flv",
    "mp3": "audio/mp3", "wav": "audio/wav", "m4a": "audio/mp4", "aac": "audio/aac", "ogg": "audio/ogg",
}
CAPTION_LANGUAGES = ["en", "en-IN", "en-US", "en-GB", "hi"]
CAPTION_NOISE = re.compile(r"\[(?:music|applause|laughter)\]\s*", re.IGNORECASE)
YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
HTTP_TIMEOUT = 15  # seconds, for the title and date lookups
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


class VideoReviewError(Exception):
    """Raised with a user-friendly message when a video cannot be turned into a review."""


def parse_youtube_id(url: str) -> str | None:
    """Return the 11-character video ID from any common YouTube link (or a bare ID), else None."""
    url = url.strip()
    if YOUTUBE_ID.match(url):
        return url
    if not re.match(r"^https?://", url):
        url = "https://" + url
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in ("youtube.com", "music.youtube.com", "youtube-nocookie.com"):
        if parsed.path == "/watch":
            candidate = urllib.parse.parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = parsed.path.strip("/").split("/")
            candidate = parts[1] if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v") else ""
    else:
        return None
    return candidate if YOUTUBE_ID.match(candidate) else None


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_youtube_details(video_id: str) -> dict:
    """Best-effort title, channel and upload date. Missing values are None; never raises."""
    details = {"title": None, "channel": None, "upload_date": None, "views": None}
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        info = json.loads(_get("https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(watch_url)))
        details.update(title=info.get("title"), channel=info.get("author_name"))
    except Exception:
        pass
    try:
        page = _get(watch_url)
        match = re.search(r'"(?:uploadDate|publishDate)":"([^"]+)"', page)
        if match:
            details["upload_date"] = pd.Timestamp(match.group(1)[:10])
        views = re.search(r'"viewCount":"(\d+)"', page)
        if views:
            details["views"] = int(views.group(1))
    except Exception:
        pass
    return details


def fetch_youtube_captions(video_id: str) -> tuple[str, list[list[float]]]:
    """Return (caption text, timestamps), preferring English/Hindi and human-made captions.

    `timestamps` is a list of [character offset in the text, seconds into the video], one per
    caption line, so any sentence in the text can be linked to the moment it is said.
    Raises VideoReviewError when there are no usable captions.
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import CouldNotRetrieveTranscript

    api = YouTubeTranscriptApi()
    try:
        available = api.list(video_id)
        try:
            transcript = available.find_transcript(CAPTION_LANGUAGES)
        except CouldNotRetrieveTranscript:
            transcript = next(iter(available))  # any language: the analysis handles Hindi/Hinglish too
        snippets = transcript.fetch()
    except (CouldNotRetrieveTranscript, StopIteration) as exc:
        raise VideoReviewError(f"No captions are available for this video ({type(exc).__name__}).") from exc
    except Exception as exc:  # network problems
        raise VideoReviewError(f"Could not download captions: {exc}") from exc
    parts, timestamps, offset = [], [], 0
    for snippet in snippets:
        piece = CAPTION_NOISE.sub("", " ".join(snippet.text.split())).strip()
        if not piece:
            continue
        timestamps.append([offset, round(float(snippet.start), 1)])
        parts.append(piece)
        offset += len(piece) + 1
    if not parts:
        raise VideoReviewError("The captions for this video are empty.")
    return " ".join(parts), timestamps


def estimate_rating(text: str) -> int:
    """Map the transcript's sentiment (-1 to 1) onto 1-5 stars."""
    return int(min(5, max(1, round(3 + 2 * score_text(text)))))


def review_id(source: str, key: str) -> str:
    """Stable ID for a video, so adding the same video twice can be detected before transcribing it."""
    prefix = "YT" if source == YOUTUBE_SOURCE else "VID"
    return f"{prefix}-{hashlib.sha1(key.encode()).hexdigest()[:8].upper()}"


def youtube_key(video_id: str) -> str:
    return f"youtube:{video_id}"


def upload_key(data: bytes) -> str:
    return "upload:" + hashlib.sha1(data).hexdigest()


def thumbnail_url(video_id: str) -> str:
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def make_review(source: str, transcript: str, product_name: str, review_date, rating: int | None,
                video_title: str, video_url: str, key: str, transcript_method: str, **extra) -> dict:
    """Build one review row. `key` identifies the video (see youtube_key and upload_key).

    `extra` holds display-only details (video_id, channel, timestamps) that are not analysed.
    """
    return {
        "review_id": review_id(source, key),
        "product_name": (product_name or "").strip() or video_title or "Unknown product",
        "rating": int(rating) if rating else estimate_rating(transcript),
        "review_text": transcript,
        "review_date": pd.Timestamp(review_date).normalize(),
        "source": source,
        "rating_estimated": not rating,
        "video_title": video_title,
        "video_url": video_url,
        "transcript_method": transcript_method,
        "video_id": "", "channel": "", "timestamps": [], "views": None,
        **extra,
    }


def _no_step(message: str) -> None:
    pass


def youtube_review(url: str, product_name: str = "", rating: int | None = None, review_date=None,
                   step=_no_step) -> dict:
    """Turn a YouTube link into a review row. review_date None means the video's upload date (or today).

    `step` is called with a short message before each stage, for progress displays.
    """
    video_id = parse_youtube_id(url)
    if not video_id:
        raise VideoReviewError(f"“{url.strip()}” is not a YouTube video link.")
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    step("Looking up the video's title and upload date")
    details = fetch_youtube_details(video_id)
    timestamps: list = []
    try:
        step("Downloading captions")
        (transcript, timestamps), method = fetch_youtube_captions(video_id), "YouTube captions"
    except VideoReviewError as caption_error:
        if not ai_service.can_transcribe_video():
            raise VideoReviewError(f"{caption_error} {ai_service.VIDEO_NEEDS_GEMINI_MESSAGE}") from caption_error
        step("No captions, so Gemini is watching and transcribing the video")
        try:
            transcript, method = ai_service.transcribe_youtube_video(watch_url), "Gemini transcription"
        except ai_service.AIServiceError as exc:
            raise VideoReviewError(f"{caption_error} Gemini could not transcribe it either: {exc}") from exc
    title = details["title"] or f"YouTube video {video_id}"
    date = review_date or details["upload_date"] or pd.Timestamp.today()
    return make_review(YOUTUBE_SOURCE, transcript, product_name, date, rating, title, watch_url,
                       youtube_key(video_id), method, video_id=video_id, channel=details["channel"] or "",
                       timestamps=timestamps, views=details.get("views"))


def uploaded_video_review(data: bytes, file_name: str, product_name: str, rating: int | None = None,
                          review_date=None, step=_no_step) -> dict:
    """Transcribe an uploaded video or audio file with Gemini and turn it into a review row."""
    extension = Path(file_name).suffix.lower().lstrip(".")
    mime_type = VIDEO_MIME_TYPES.get(extension)
    if not mime_type:
        raise VideoReviewError(f"{file_name}: unsupported file type. Use one of: {', '.join(VIDEO_MIME_TYPES)}.")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"video.{extension}"
        path.write_bytes(data)
        step("Uploading to Gemini and transcribing the speech")
        try:
            transcript = ai_service.transcribe_video_file(str(path), mime_type)
        except ai_service.AIServiceError as exc:
            raise VideoReviewError(f"{file_name}: {exc}") from exc
    return make_review(UPLOAD_SOURCE, transcript, product_name, review_date or pd.Timestamp.today(), rating,
                       Path(file_name).stem, "", upload_key(data), "Gemini transcription")


def videos_to_frame(videos: list[dict]) -> pd.DataFrame:
    """Review rows for the added videos, in the standard review schema plus the video columns."""
    frame = pd.DataFrame(videos, columns=VIDEO_COLUMNS)
    frame["review_date"] = pd.to_datetime(frame["review_date"])
    frame["rating"] = pd.to_numeric(frame["rating"])
    return frame


def analysis_fields(videos: list[dict]) -> list[dict]:
    """Only the fields the analysis uses (keeps the cache key small: no timestamps)."""
    return [{column: video.get(column) for column in VIDEO_COLUMNS} for video in videos]


# ---------------------------------------------------------------- saving between sessions

STORE_PATH = Path(__file__).resolve().parent / ".productpulse" / "video_reviews.json"


def load_saved(path: Path | None = None) -> list[dict]:
    """Video reviews saved by earlier sessions (an empty list if there are none or the file is unreadable)."""
    path = path or STORE_PATH
    try:
        videos = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [v for v in videos if isinstance(v, dict) and {"review_id", "review_text"} <= set(v)]


def save(videos: list[dict], path: Path | None = None) -> bool:
    """Save the video reviews so they come back next time. Returns False if the file cannot be written."""
    path = path or STORE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(videos, default=str, ensure_ascii=False, indent=1), encoding="utf-8")
        return True
    except OSError:
        return False


def format_views(views) -> str:
    """1234567 -> "1.2M views"."""
    if not views:
        return ""
    views = int(views)
    for size, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if views >= size:
            return f"{views / size:.1f}".rstrip("0").rstrip(".") + f"{suffix} views"
    return f"{views} views"


# ---------------------------------------------------------------- moments in a video

def moment_seconds(video: dict, clause: str) -> float | None:
    """Seconds into the video where `clause` is said, or None if it cannot be located."""
    timestamps = video.get("timestamps") or []
    position = str(video.get("review_text", "")).find(str(clause).strip())
    if position < 0 or not timestamps:
        return None
    seconds = timestamps[0][1]
    for offset, start in timestamps:
        if offset > position:
            break
        seconds = start
    return seconds


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


def moment_link(video: dict, clause: str) -> tuple[str, str] | None:
    """(URL that starts the YouTube video at the clause, "m:ss" label), or None."""
    seconds = moment_seconds(video, clause)
    if seconds is None or not video.get("video_id"):
        return None
    return f"https://www.youtube.com/watch?v={video['video_id']}&t={int(seconds)}s", format_seconds(seconds)

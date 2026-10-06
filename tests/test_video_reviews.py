"""Video reviews: link parsing, review rows, fallbacks and merging into the analysis (no network or API calls)."""

import json

import pandas as pd
import pytest

import ai_service
import app
import video_reviews
from video_reviews import VideoReviewError

VIDEO_ID = "dQw4w9WgXcQ"
TRANSCRIPT = ("Hi everyone, today I am reviewing these headphones. The battery life is amazing, easily two days. "
              "But the bluetooth keeps disconnecting from my laptop, which is really annoying.")


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VIDEO_ID}",
    f"https://youtube.com/watch?v={VIDEO_ID}&t=42s",
    f"https://m.youtube.com/watch?feature=share&v={VIDEO_ID}",
    f"https://youtu.be/{VIDEO_ID}?si=abc",
    f"https://www.youtube.com/shorts/{VIDEO_ID}",
    f"https://www.youtube.com/embed/{VIDEO_ID}",
    f"youtube.com/live/{VIDEO_ID}",
    VIDEO_ID,
])
def test_parse_youtube_id(url):
    assert video_reviews.parse_youtube_id(url) == VIDEO_ID


@pytest.mark.parametrize("url", ["https://vimeo.com/123456", "https://www.youtube.com/@channel", "not a link", ""])
def test_parse_youtube_id_rejects_other_links(url):
    assert video_reviews.parse_youtube_id(url) is None


def test_estimated_rating_follows_sentiment():
    assert video_reviews.estimate_rating("This is excellent, I love it, best purchase ever!") >= 4
    assert video_reviews.estimate_rating("Terrible. It broke in a day, worst product, total waste of money.") <= 2


@pytest.fixture
def offline(monkeypatch):
    """No network: fixed video details, and captions that tests can switch off."""
    state = {"captions": TRANSCRIPT}

    def captions(video_id):
        if state["captions"] is None:
            raise VideoReviewError("No captions are available for this video.")
        text = state["captions"]
        # Two caption lines: the second sentence starts 42.5 seconds in.
        second = text.index("But the bluetooth") if "But the bluetooth" in text else len(text)
        return text, [[0, 1.0], [second, 42.5]]

    monkeypatch.setattr(video_reviews, "fetch_youtube_details", lambda video_id: {
        "title": "Nimbus headphones review", "channel": "Tech", "upload_date": pd.Timestamp("2026-09-15")})
    monkeypatch.setattr(video_reviews, "fetch_youtube_captions", captions)
    return state


def test_youtube_review_uses_captions_and_upload_date(offline):
    review = video_reviews.youtube_review(f"https://youtu.be/{VIDEO_ID}")
    assert review["review_text"] == TRANSCRIPT
    assert review["transcript_method"] == "YouTube captions"
    assert review["product_name"] == "Nimbus headphones review"  # video title when no product is given
    assert review["review_date"] == pd.Timestamp("2026-09-15")
    assert review["rating_estimated"] and 1 <= review["rating"] <= 5
    assert review["source"] == video_reviews.YOUTUBE_SOURCE
    assert review["review_id"] == video_reviews.review_id(video_reviews.YOUTUBE_SOURCE, video_reviews.youtube_key(VIDEO_ID))


def test_youtube_review_keeps_user_choices(offline):
    review = video_reviews.youtube_review(VIDEO_ID, "Nimbus ANC Headphones", 2, pd.Timestamp("2026-10-01"))
    assert review["product_name"] == "Nimbus ANC Headphones"
    assert review["rating"] == 2 and not review["rating_estimated"]
    assert review["review_date"] == pd.Timestamp("2026-10-01")


def test_youtube_without_captions_needs_gemini(offline):
    offline["captions"] = None
    with pytest.raises(VideoReviewError, match="Gemini"):
        video_reviews.youtube_review(VIDEO_ID)


def test_youtube_without_captions_falls_back_to_gemini(offline, monkeypatch):
    offline["captions"] = None
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("AI_API_KEY", "test-key")
    sent = []

    def fake_call(system, prompt, max_tokens=2000, media=None):
        sent.append(media)
        return TRANSCRIPT

    monkeypatch.setattr(ai_service, "_call_model", fake_call)
    review = video_reviews.youtube_review(VIDEO_ID)
    assert review["transcript_method"] == "Gemini transcription"
    assert sent[0][0].file_data.file_uri == f"https://www.youtube.com/watch?v={VIDEO_ID}"


def test_no_speech_is_an_error(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("AI_API_KEY", "test-key")
    monkeypatch.setattr(ai_service, "_call_model", lambda *args, **kwargs: "NO_SPEECH")
    with pytest.raises(ai_service.AIServiceError, match="No speech"):
        ai_service.transcribe_youtube_video(f"https://www.youtube.com/watch?v={VIDEO_ID}")


def test_video_transcription_needs_gemini(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "anthropic")
    monkeypatch.setenv("AI_API_KEY", "sk-ant-test")
    assert not ai_service.can_transcribe_video()
    with pytest.raises(ai_service.AIServiceError, match="Gemini"):
        ai_service.transcribe_video_file("video.mp4", "video/mp4")


def test_upload_rejects_unknown_file_type():
    with pytest.raises(VideoReviewError, match="unsupported"):
        video_reviews.uploaded_video_review(b"data", "notes.txt", "Product")


def test_uploaded_video_review(monkeypatch):
    monkeypatch.setattr(ai_service, "transcribe_video_file", lambda path, mime: TRANSCRIPT)
    review = video_reviews.uploaded_video_review(b"fake video", "my review.mp4", "Nimbus ANC Headphones", 4)
    assert review["source"] == video_reviews.UPLOAD_SOURCE
    assert review["video_title"] == "my review" and review["rating"] == 4
    assert review["review_id"] == video_reviews.review_id(video_reviews.UPLOAD_SOURCE, video_reviews.upload_key(b"fake video"))


def test_video_reviews_join_the_analysis(offline):
    review = video_reviews.youtube_review(VIDEO_ID, "Nimbus ANC Headphones")
    sample = str(app.DATASETS["Demo · Mixed products (24 reviews)"])
    analysis = app.analyse_dataset.__wrapped__(None, None, sample, json.dumps([review], default=str))
    df = analysis["dataframe"]
    assert len(df) == 25 and "+ 1 video review(s)" in analysis["source_label"]
    video = df[df["review_id"] == review["review_id"]].iloc[0]
    assert video["source"] == video_reviews.YOUTUBE_SOURCE and video["sentiment"] in ("Positive", "Neutral", "Negative")
    assert set(df["source"]) == {ai_service.TEXT_SOURCE, video_reviews.YOUTUBE_SOURCE}
    themes = set(analysis["mentions"].loc[analysis["mentions"]["review_id"] == review["review_id"], "theme"])
    assert themes  # battery / bluetooth are picked up from the transcript


def test_transcript_gets_more_room_in_ai_prompts():
    long_text = "word " * 3000
    row = pd.Series({"review_id": "YT-1", "rating": 4, "review_date": "2026-09-01", "product_name": "P",
                     "review_text": long_text, "source": video_reviews.YOUTUBE_SOURCE, "rating_estimated": True})
    line = ai_service.format_review_line(row)
    assert "video transcript" in line and "(estimated)" in line
    assert len(line) > ai_service.MAX_REVIEW_CHARS * 5
    assert len(ai_service.format_review_line(row.drop(["source", "rating_estimated"]))) < ai_service.MAX_REVIEW_CHARS + 100


def test_empty_transcript_is_retried_once(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("AI_API_KEY", "test-key")
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ai_service.AIServiceError(ai_service.EMPTY_RESPONSE_MESSAGE)
        return TRANSCRIPT

    monkeypatch.setattr(ai_service, "_call_model", flaky)
    assert ai_service.transcribe_youtube_video(f"https://www.youtube.com/watch?v={VIDEO_ID}") == TRANSCRIPT
    assert len(calls) == 2


def test_moment_links_point_to_the_second_a_clause_is_said(offline):
    review = video_reviews.youtube_review(VIDEO_ID)
    assert review["timestamps"] and review["video_id"] == VIDEO_ID
    url, label = video_reviews.moment_link(review, "But the bluetooth keeps disconnecting from my laptop")
    assert url.endswith(f"v={VIDEO_ID}&t=42s") and label == "0:42"
    assert video_reviews.moment_link(review, "The battery life is amazing")[1] == "0:01"
    assert video_reviews.moment_link(review, "words that are not in the transcript") is None
    assert video_reviews.format_seconds(3725) == "1:02:05"


def test_uploaded_videos_have_no_moment_links(monkeypatch):
    monkeypatch.setattr(ai_service, "transcribe_video_file", lambda path, mime: TRANSCRIPT)
    review = video_reviews.uploaded_video_review(b"fake video", "clip.mp4", "Product")
    assert video_reviews.moment_link(review, "The battery life is amazing") is None


def test_analysis_fields_leave_out_display_details(offline):
    review = video_reviews.youtube_review(VIDEO_ID)
    fields = video_reviews.analysis_fields([review])[0]
    assert set(fields) == set(video_reviews.VIDEO_COLUMNS) and "timestamps" not in fields

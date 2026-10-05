"""AI service: configuration, prompts, chunking, citation checks and error handling (no real API calls)."""

import pytest

import ai_service

FAKE_ANSWER = "### Answer\nCustomers mention delivery. [R003, R012, R999]"


@pytest.fixture
def capture(monkeypatch):
    """Replace the model call and record every prompt sent."""
    prompts = []

    def fake_call(system, prompt, max_tokens=2000):
        prompts.append({"system": system, "prompt": prompt})
        return FAKE_ANSWER

    monkeypatch.setattr(ai_service, "_call_model", fake_call)
    return prompts


def test_no_api_key_gives_friendly_error(sample_reviews):
    assert not ai_service.is_ai_configured()
    with pytest.raises(ai_service.AIServiceError, match="AI_API_KEY"):
        ai_service.generate_review_summary(sample_reviews)


def test_model_name_is_configurable(monkeypatch):
    monkeypatch.delenv("AI_MODEL", raising=False)
    assert ai_service.get_model_name() == ai_service.DEFAULT_MODEL
    monkeypatch.setenv("AI_MODEL", "my-model")
    assert ai_service.get_model_name() == "my-model"


def test_summary_single_request_and_citation_check(sample_reviews, capture):
    result = ai_service.generate_review_summary(sample_reviews, "all products")
    assert len(capture) == 1 and result["chunks"] == 1
    assert "[R001]" in capture[0]["prompt"]  # real review text is sent with IDs
    assert "Only" in capture[0]["system"] or "ONLY" in capture[0]["system"]
    assert result["text"] == result["summary"] == FAKE_ANSWER
    assert result["cited_ids"] == ["R003", "R012"]
    assert result["unknown_ids"] == ["R999"]


def test_positive_and_negative_summaries(sample_reviews, capture):
    positive = sample_reviews[sample_reviews["sentiment"] == "Positive"]
    negative = sample_reviews[sample_reviews["sentiment"] == "Negative"]
    ai_service.generate_positive_summary(positive, "all products")
    ai_service.generate_negative_summary(negative, "all products")
    assert "POSITIVE reviews" in capture[0]["prompt"]
    assert "NEGATIVE reviews" in capture[1]["prompt"]
    # Only the positive reviews are sent for the positive summary.
    for review_id in negative["review_id"]:
        assert f"[{review_id}]" not in capture[0]["prompt"]


def test_empty_groups_raise_friendly_errors(sample_reviews, capture):
    with pytest.raises(ai_service.AIServiceError, match="no positive reviews"):
        ai_service.generate_positive_summary(sample_reviews.iloc[0:0])
    with pytest.raises(ai_service.AIServiceError, match="no negative reviews"):
        ai_service.generate_negative_summary(sample_reviews.iloc[0:0])
    assert capture == []


def test_key_insights_include_statistics(sample_reviews, capture):
    ai_service.generate_key_insights(sample_reviews, "all products", stats="Average rating: 3.08/5")
    prompt = capture[0]["prompt"]
    assert "<statistics>" in prompt and "Average rating: 3.08/5" in prompt
    assert "Recommended Actions" in prompt


def test_answer_question_grounded(sample_reviews, capture):
    result = ai_service.answer_question("What are the biggest complaints?", sample_reviews.head(5),
                                        stats="Reviews: 24", subject="all products")
    prompt = capture[0]["prompt"]
    assert "<question>\nWhat are the biggest complaints?\n</question>" in prompt
    assert "Reviews: 24" in prompt
    # Cited IDs are validated against the reviews actually sent.
    sent = set(sample_reviews.head(5)["review_id"])
    assert set(result["cited_ids"]) <= sent
    assert "R999" in result["unknown_ids"]


def test_empty_or_long_question_rejected(sample_reviews, capture):
    with pytest.raises(ai_service.AIServiceError, match="type a question"):
        ai_service.answer_question("   ", sample_reviews, stats="")
    with pytest.raises(ai_service.AIServiceError, match="500 characters"):
        ai_service.answer_question("x" * 501, sample_reviews, stats="")
    assert capture == []


def test_large_dataset_is_chunked(monkeypatch, sample_reviews):
    calls = []
    monkeypatch.setattr(ai_service, "CHUNK_CHAR_LIMIT", 800)
    monkeypatch.setattr(
        ai_service, "_call_model", lambda system, prompt, max_tokens=2000: calls.append(prompt) or "- note [R001]"
    )
    result = ai_service.generate_review_summary(sample_reviews)
    assert result["chunks"] > 1
    assert len(calls) == result["chunks"] + 1  # one per chunk + one combine step
    assert "<batch_notes>" in calls[-1]


def test_extract_cited_ids_ignores_prose_brackets():
    known, unknown = ai_service.extract_cited_ids("See [R001, R002] and [cited review IDs]", {"R001", "R002"})
    assert known == ["R001", "R002"]
    assert unknown == []


def test_api_errors_become_friendly(monkeypatch, sample_reviews):
    import anthropic
    import httpx2

    monkeypatch.setenv("AI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("AI_PROVIDER", "anthropic")
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def raise_connection_error(self, **kwargs):
        raise anthropic.APIConnectionError(request=request)

    monkeypatch.setattr(anthropic.resources.Messages, "create", raise_connection_error)
    with pytest.raises(ai_service.AIServiceError, match="Could not reach"):
        ai_service.generate_review_summary(sample_reviews)


# ---------- provider selection and Gemini ----------

def test_provider_is_detected_from_key_or_setting(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "sk-ant-not-real")
    assert ai_service.get_provider() == "anthropic"
    assert ai_service.get_model_name() == "claude-haiku-4-5"
    monkeypatch.setenv("AI_API_KEY", "AIza-not-real")
    assert ai_service.get_provider() == "gemini"
    assert ai_service.get_model_name() == "gemini-flash-latest"
    monkeypatch.setenv("AI_PROVIDER", "anthropic")
    assert ai_service.get_provider() == "anthropic"
    assert "Claude" in ai_service.provider_label()


class _FakeResponse:
    def __init__(self, text):
        self.text = text
        self.candidates = []


def _fake_gemini(monkeypatch, behaviour):
    """Patch the google-genai client so generate_content runs `behaviour(model)`."""
    from google import genai

    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.append(model)
            return behaviour(model)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setenv("AI_API_KEY", "AIza-not-real")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    return calls


def test_gemini_overload_falls_back_to_lite_model(monkeypatch, sample_reviews):
    from google.genai import errors

    def behaviour(model):
        if model == "gemini-flash-latest":
            raise errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})
        return _FakeResponse("### Overall Summary\nFine. [R001]")

    calls = _fake_gemini(monkeypatch, behaviour)
    result = ai_service.generate_review_summary(sample_reviews)
    assert calls == ["gemini-flash-latest", ai_service.GEMINI_FALLBACK_MODEL]
    assert result["cited_ids"] == ["R001"]


def test_gemini_rate_limit_gives_friendly_error(monkeypatch, sample_reviews):
    from google.genai import errors

    def behaviour(model):
        raise errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    _fake_gemini(monkeypatch, behaviour)
    with pytest.raises(ai_service.AIServiceError, match="free-tier limit"):
        ai_service.generate_review_summary(sample_reviews)

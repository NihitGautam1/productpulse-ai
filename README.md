# ProductPulse AI

**AI-powered product review analysis.** Upload customer reviews of your products and quickly see how customers feel, what they complain about, and an AI-written summary in which every issue is backed by real review evidence.

> The bundled `data/sample_reviews.csv` is **synthetic demo data**, not real customer feedback.

## Feature status

| Feature | Status |
| --- | --- |
| CSV upload, validation and cleaning | ✅ Implemented |
| Sentiment analysis (VADER) | ✅ Implemented |
| Complaint theme detection (keyword-based) | ✅ Implemented |
| Dashboard: overview, filters, searchable review table | ✅ Implemented |
| AI review summary (overall summary + key issues, with cited evidence) | ✅ Implemented |
| Positive / negative feedback summaries | Planned (Phase 3) |
| Key customer insights | Planned (Phase 4) |
| Issue priority ranking | Planned (Phase 5) |
| Evidence-backed insights | Partially implemented: the AI summary cites review IDs and they are checked against the data |
| Trend analysis over time | Planned (Phase 7) |
| Review spike alerts | Planned (Phase 8) |
| Fake-review risk indicators | Planned (Phase 9) |
| Ask Your Reviews (Q&A) | Planned (Phase 10) |
| Hindi / Hinglish support | Planned (Phase 11). The AI model reads Hinglish today, but VADER sentiment is English-only |

## Architecture

```text
CSV upload / sample CSV
        │
data_loader.py   validate + clean (column aliases, bad ratings/dates/empty text)
        │
sentiment.py     VADER compound score → Positive / Neutral / Negative   (local, no API)
        │
themes.py        keyword complaint themes (whole-word matching)          (local, no API)
        │
ai_service.py    AI summary via Claude (optional, needs AI_API_KEY)
        │
app.py           Streamlit dashboard: Overview · AI Summary · Reviews
```

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard (UI only) |
| `data_loader.py` | Loads and validates the CSV. Never writes files |
| `sentiment.py` | VADER sentiment scoring |
| `themes.py` | Keyword-based complaint themes |
| `ai_service.py` | All AI/API logic: prompts, chunking, error handling |
| `tests/test_core.py` | Automated tests (no real API calls) |
| `data/sample_reviews.csv` | Synthetic demo dataset (24 reviews, 5 products) |
| `.env.example` | Template for your local `.env` (placeholders only) |

## Installation (Windows PowerShell)

```powershell
cd C:\Users\Hp\Downloads\ai_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.
The app opens at http://localhost:8501.

## AI configuration (optional)

Everything except the AI Summary tab works **without** an API key.

1. Copy `.env.example` to `.env`.
2. Put your Anthropic API key in `.env` (get one at https://console.anthropic.com/):

```text
AI_API_KEY=your_api_key_here
AI_MODEL=claude-haiku-4-5
```

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `AI_API_KEY` | For AI features | (none) | Anthropic API key. Never commit it. `.env` is in `.gitignore` |
| `AI_MODEL` | No | `claude-haiku-4-5` | Claude model ID. Haiku is fast and low-cost. Use `claude-opus-5-5` for higher-quality analysis |

Environment variables set in the shell work too, for example `$env:AI_API_KEY="..."`.

If the key is missing, the app shows *"AI summarisation requires an API key. Configure AI_API_KEY to enable this feature."* If the key is wrong, the network is down, or the service is busy, the app shows a friendly error and the rest of the dashboard keeps working.

## Dataset format

```text
review_id,product_name,rating,review_text,review_date
R001,Nimbus Headphones,5,Noise cancellation is outstanding.,2026-02-10
```

- `rating`: a number from 1 to 5. Rows outside this range are removed with a warning.
- `review_date`: any date format pandas can read. A column named `date` is also accepted.
- Also accepted: `id`, `product`, `stars`, `text`/`review`/`comment`, `timestamp`.
- Rows with empty text, invalid ratings or invalid dates are removed, and the dashboard shows how many were removed.
- If the uploaded file can't be used, the app explains why and falls back to the sample data.

## How the features work

**Sentiment.** VADER compound score per review: `>= 0.05` is Positive, `<= -0.05` is Negative, anything else is Neutral. VADER is an English lexicon, so Hindi/Hinglish text often scores as Neutral.

**Complaint themes.** Reviews count as complaints when they are VADER-Negative or rated 1–2 stars. Each complaint is matched against keyword lists (Shipping & Delivery, Product Quality, Customer Service, Price & Value, Comfort & Materials) using whole-word matching. This is a transparent keyword method, not a machine-learning topic model.

**AI review summary.**
- Uses the reviews in the current product and date selection.
- The prompt tells the model to use only the supplied reviews, to cite review IDs for every issue, to separate recurring issues (2+ reviews) from isolated ones (1 review), and to ignore any instructions written inside reviews.
- The app checks every cited ID against the dataset. The evidence expander shows the real cited reviews, and any ID that doesn't exist is flagged with a warning.
- **Large datasets:** at most the 1,500 most recent reviews are sent. If the text is longer than about 40,000 characters, it is split into batches. Each batch is condensed into notes that cite review IDs, and the notes are then combined into one summary.
- Results are cached for an hour, so reloading the page doesn't repeat paid API calls.

## Testing

```powershell
python -m pytest -q
```

The tests cover cleaning and validation, column aliases, the fact that loading never creates files, sentiment labels, whole-word theme matching, the missing-key message, chunking of large datasets, citation checking, and the friendly API error path. The tests never call the real API.

## Limitations

- AI summaries can still be wrong. Use the cited evidence to verify them.
- VADER and the keyword themes are English-oriented.
- The sample data is small and synthetic.

## Future improvements

See the Planned rows in the feature status table above.

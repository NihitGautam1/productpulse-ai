# ProductPulse AI

**AI-powered product review intelligence.** ProductPulse AI turns hundreds or thousands of customer reviews into insights a business can act on. It shows how customers feel, what they like and dislike, which problems matter most, whether problems are growing, which reviews look suspicious, and an AI summary in which every claim links back to real reviews.

It supports **English, Hindi and Hinglish** reviews.

> **About the data:** both bundled datasets are **synthetic**. They were made up for demonstration and testing, and they are **not real customer feedback**.
> - `data/demo_electronics_reviews.csv`: 295 reviews of 3 products over 6 months, produced by `scripts/generate_demo_data.py`. It deliberately includes Hinglish/Hindi reviews, a September rise in Bluetooth complaints, and a burst of near-identical 5-star reviews.
> - `data/sample_reviews.csv`: 24 short reviews of 5 products.

---

## Features

| # | Feature | Status | How it works |
| --- | --- | --- | --- |
| 1 | CSV upload, validation and cleaning | IMPLEMENTED | `data_loader.py` |
| 2 | Sentiment analysis | IMPLEMENTED | VADER, extended for review vocabulary and Hinglish/Hindi |
| 3 | Complaint and praise theme detection | IMPLEMENTED | Clause-level keyword matching (`themes.py`) |
| 4 | AI review summary | IMPLEMENTED | Gemini or Claude (`ai_service.py`) |
| 5 | Positive and negative feedback summaries | IMPLEMENTED | AI, run on the VADER-positive and VADER-negative groups |
| 6 | Key customer insights and recommended actions | IMPLEMENTED | Computed from the data, plus optional AI insights |
| 7 | Issue priority ranking | IMPLEMENTED | Transparent 0–100 score (`insights.py`) |
| 8 | Evidence-backed insights | IMPLEMENTED | Real review quotes; AI-cited IDs are checked against the data |
| 9 | Trend analysis over time | IMPLEMENTED | `trends.py` |
| 10 | Review spike alerts | IMPLEMENTED | Recent window vs baseline, with minimum-data rules |
| 11 | Fake-review risk indicators | IMPLEMENTED | Explainable signals (`fake_risk.py`) |
| 12 | Ask Your Reviews (Q&A) | IMPLEMENTED | TF-IDF retrieval plus grounded AI answer |
| 13 | Hindi / Hinglish support | IMPLEMENTED (limited accuracy, see Limitations) | `sentiment.py`, `themes.py` |

All AI features are optional. **AI output quality depends on the model. The AI features were verified with mocked responses in the automated tests.** Everything that doesn't use AI works with no API key.

---

## Dashboard pages

| Page | What it shows |
| --- | --- |
| **Overview** | Key numbers (reviews, average rating, positive/negative share, health score) with mini trend charts, active alerts, sentiment mix, rating distribution, top priority issues with an example quote, most-praised features, key insights, and the AI summary |
| **AI Insights** | AI overall summary, positive feedback, negative feedback, key insights and recommended actions, each with a panel listing the cited reviews. Data-driven insights are always shown, even without AI |
| **Ask Your Reviews** | Ask questions in plain language. Answers come from the most relevant reviews and computed statistics, with the cited reviews shown |
| **Priority Issues** | Ranked issues with their score breakdown, praise vs complaints per theme, and real review quotes as evidence for each issue |
| **Trends & Alerts** | Review volume, negative share, average rating and complaints per theme over time, plus spike alerts |
| **Review Integrity** | Fake-review risk levels with the reasons for each flag, and a clear disclaimer |
| **Review Explorer** | All reviews with search and filters (sentiment, language, risk, rating, theme), a CSV download of the enriched data, and the language mix |

Pages are reached from the navigation bar at the top, grouped as Dashboard, AI analysis, Issues, and Trust & data. The sidebar holds the dataset choice (demo data or your own upload), the product filter, the date range and the AI status. The light and dark themes both have their own colour sets.

---

## Architecture

```text
            CSV upload  /  bundled synthetic demo data
                              │
                       data_loader.py      validate + clean (aliases, ratings, dates, empty text)
                              │
                       sentiment.py        VADER score + label, language (English/Hinglish/Hindi)
                              │
                       themes.py           clause-level praise / complaint per theme
                              │
     ┌──────────────┬─────────┼───────────────┬────────────────┐
 insights.py     trends.py   fake_risk.py   retrieval.py     ai_service.py
 priority,       trends,     risk score +   relevant reviews  Gemini/Claude: summaries,
 key insights,   spike       reasons        for questions     insights, Q&A
 evidence        alerts                                       (optional)
     └──────────────┴─────────┼───────────────┴────────────────┘
                              │
          app.py (navigation, sidebar, filters, caching)
          views.py (pages) · charts.py (Plotly) · styles.py (CSS)
```

Everything except `ai_service.py` runs locally on the CPU with no API key.

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit entry point: page navigation, sidebar (dataset, filters, AI status), shared cached data |
| `views.py` | One function per dashboard page |
| `charts.py` | Plotly chart builders with consistent colours (blue = positive, grey = neutral, red = negative) |
| `styles.py` | Custom CSS for cards, badges and layout |
| `data_loader.py` | Loads and validates CSV data. Never writes files |
| `sentiment.py` | VADER sentiment with a review vocabulary and Hinglish/Hindi extensions, plus language detection |
| `themes.py` | Theme keywords; splits reviews into clauses and labels each theme mention as praise or complaint |
| `insights.py` | Priority ranking, data-driven key insights, evidence selection, health score, statistics for the AI |
| `trends.py` | Time series and spike alerts |
| `fake_risk.py` | Fake-review risk score and reasons |
| `retrieval.py` | Finds the reviews most relevant to a question (TF-IDF plus theme matching) |
| `ai_service.py` | All AI/API code: prompts, grounding rules, chunking, citation checking, friendly errors |
| `scripts/generate_demo_data.py` | Rebuilds the synthetic demo dataset (fixed random seed) |
| `tests/` | Automated tests (pytest). They never call the real API |
| `.streamlit/config.toml` | Theme (light and dark), fonts, upload limit |
| `.env.example` | Template for your local `.env` (placeholders only) |

---

## Installation (Windows PowerShell)

```powershell
cd C:\Users\Hp\Downloads\ai_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

The app opens at **http://localhost:8501**. Press `Ctrl+C` in PowerShell to stop it.

Requirements: Python 3.10 or newer. Internet access is needed only to install packages and for the optional AI features.

---

## AI configuration (optional)

ProductPulse AI supports two AI providers: **Google Gemini** (has a free tier) and **Anthropic Claude**.

1. Copy `.env.example` to `.env`. `.env` is in `.gitignore` and must never be committed.
2. Add your key. For example, with a free Gemini key from https://aistudio.google.com/apikey:

```text
AI_PROVIDER=gemini
AI_API_KEY=your_api_key_here
```

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `AI_API_KEY` | For AI features | (none) | Gemini or Anthropic API key. It is never shown in the UI or logs |
| `AI_PROVIDER` | No | guessed from the key | `gemini` or `anthropic`. If not set, keys starting with `sk-ant-` use Claude and any other key uses Gemini |
| `AI_MODEL` | No | `gemini-flash-latest` / `claude-haiku-4-5` | Model ID. `gemini-flash-latest` always points to Google's current Flash model. For Claude, `claude-opus-5-5` gives the highest-quality analysis |

**Gemini free tier:** models are sometimes briefly overloaded (HTTP 503) or rate-limited (HTTP 429). When that happens, the app automatically retries once with `gemini-flash-lite-latest`. If that also fails, it shows a friendly "try again" message.

**Verified with a real Gemini key:**
- The overall, positive and negative summaries, the insights and Q&A were all generated successfully.
- Every cited review ID was valid.
- "Are delivery complaints increasing?" was answered correctly ("no": 9.3% recently vs 13.2% earlier) using the computed statistics.

Shell variables work too, for example `$env:AI_API_KEY="..."`.

**Without a key**, AI sections show *"AI summarisation requires an API key. Configure AI_API_KEY to enable this feature."*, and everything else keeps working. If the key is wrong, the network is down, the model name is invalid or you hit a rate limit, the app shows a friendly message instead of crashing.

---

## Dataset format

```text
review_id,product_name,rating,review_text,review_date
R001,Nimbus Headphones,5,Noise cancellation is outstanding.,2026-02-10
```

| Column | Rules | Also accepted as |
| --- | --- | --- |
| `review_id` | Any text. Missing IDs are generated | `id`, `reviewid` |
| `product_name` | Any text | `product`, `productname` |
| `rating` | Number from 1 to 5. Other values are removed with a warning | `stars`, `score` |
| `review_text` | Must not be empty. English, Hindi or Hinglish | `text`, `review`, `comment`, `reviewtext` |
| `review_date` | Any date pandas can read | `date`, `timestamp`, `reviewdate` |

Column names are case-insensitive. The dashboard reports how many rows were removed and why. If an uploaded file can't be used at all, the app explains why and falls back to the demo data.

---

## How each feature works

### Sentiment (VADER, extended)
Each review gets a VADER compound score: `>= 0.05` is Positive, `<= -0.05` is Negative, and anything in between is Neutral. VADER runs locally. It is extended with:
- **Review vocabulary VADER lacks**, e.g. "disconnecting", "cracked", "overpriced", "reliable".
- **Hinglish and Hindi words**: positive words (accha, badhiya, mast, अच्छा, शानदार), negative words (bekar, kharab, ghatiya, खराब, बेकार), intensifiers (bahut, ekdum, बहुत) and negations (nahi, नहीं).
- **Rewrites before scoring** (the text users see is never changed):
  - Hindi negation comes *after* the word it negates ("accha nahi hai"), so it is moved in front for VADER ("not accha").
  - "customer care/support/service" is treated as a topic, not praise.
  - "koi jawab nahi diya" ("gave no reply") is treated as unresponsive.

**Language detection** labels each review English, Hinglish or Hindi. Hindi means Devanagari script. Hinglish means common romanised-Hindi words such as hai, nahi, bahut, accha or kharab.

### Themes: what customers praise and complain about
There are 9 themes: Shipping & Delivery, Product Quality, Customer Service, Price & Value, Comfort & Materials, Battery & Charging, Connectivity, Performance & Features, Design & Usability.

1. Each review is split into clauses at sentence ends and contrast words (but, however, lekin, magar, लेकिन).
2. Each clause is matched against theme keywords, as whole words only, so "plate" doesn't match "late".
3. A theme mention is a **complaint** if the clause contains a complaint keyword (e.g. "disconnecting", "cracked", "late") or scores negative. It is **praise** if the clause scores positive. Unclear clauses fall back to the star rating.

So "Great value for the price. However, the left earcup randomly loses connection." counts as *praise* for Price & Value and a *complaint* about Connectivity. Phrases like "sound quality" count as Performance & Features, not Product Quality. This is a transparent keyword method, **not** a machine-learning topic model.

### Issue priority ranking
Each complaint theme gets a 0–100 score:

```text
score = 40 x frequency      share of all reviews that complain about the theme (20% or more = full marks)
      + 25 x negativity     how negative the complaint clauses are (average VADER score, 0 to 1)
      + 20 x low_ratings    share of those complaint reviews rated 1-2 stars
      + 15 x recent_rise    1 = complaint rate rose 50%+ in the recent window, 0.5 = rose, 0 = flat or fell
```

The recent window is the last 30 days, or the last quarter of the date span if the data covers less than 120 days.

Each level needs both a high enough score **and** enough evidence, so one complaint can never be "Critical":

| Level | Rule |
| --- | --- |
| Critical | score >= 70 **and** at least 5 complaint reviews |
| High | score >= 50 **and** at least 3 complaint reviews |
| Medium | score >= 30 **and** at least 2 complaint reviews |
| Low | everything else, including any issue raised by only one review |

### Key insights and evidence
**Data-driven insights** are always available, even without AI:
- overall sentiment
- the most-praised themes
- recurring complaints
- rising complaints
- the biggest driver of 1–2 star ratings
- isolated complaints that are not yet a pattern
- the language mix

Each insight lists the IDs of the real reviews that support it. **Evidence** for an issue means the actual review quotes with the strongest relevant clauses, never invented text.

### AI summaries, insights and grounding
`ai_service.py` sends the AI model only real reviews, each tagged with its ID, plus statistics computed by the app. The model is instructed to:
- use only the supplied reviews and statistics, and never invent facts
- cite review IDs for every claim
- call a point "recurring" only if two or more reviews support it, and "isolated" otherwise
- take numbers from the supplied statistics instead of counting itself
- ignore any instructions written inside reviews

The app then **checks every cited ID**. Cited reviews are shown as evidence, and any ID that doesn't exist in the data is flagged with a warning.

**Large datasets:** at most the 1,500 most recent reviews are used. If the text is longer than about 40,000 characters, it is split into batches. Each batch is condensed into notes that keep the review IDs, and the notes are combined into the final answer. Results are kept for the browser session, per selection, so moving between pages doesn't repeat paid API calls. The four AI Insights sections run in parallel.

### Ask Your Reviews
1. `retrieval.py` finds the reviews most relevant to the question using TF-IDF text similarity. Reviews that match a theme named in the question get a boost, e.g. "delivery" boosts Shipping & Delivery reviews. General questions ("what do customers like most?") get a balanced sample of the most positive, most negative and most recent reviews.
2. The AI answers using those reviews plus the computed statistics, including complaints per theme per month. That is how it can answer "Are delivery complaints increasing?" from real numbers.
3. The answer cites review IDs, and those reviews are shown below it. Without an API key, the page still shows the most relevant reviews and the statistics.

### Trends
Review volume (split by sentiment), negative share, average rating and complaints per theme are grouped daily, weekly or monthly. By default the grouping is chosen automatically from the date span. Empty periods are shown as gaps instead of being hidden. The complaint-themes chart is grouped one level coarser (for example monthly when the rest is weekly), because five weekly theme lines are too noisy to read.

### Spike alerts
The **recent window** (the last 14 days, or the last quarter of a short date span) is compared with a **baseline** (up to 8 weeks before it). The app checks for jumps in:
- negative reviews
- 1–2 star ratings
- complaints about each theme
- review volume
- average rating

**Minimum-data rules** prevent false alarms on small datasets:
- at least **30 reviews** in total
- at least **8 reviews** in the recent window
- at least **15 reviews** in the baseline

Each alert also needs a minimum number of affected reviews (4 for negative reviews and low ratings, 3 for theme complaints) and a clear jump: at least 1.5 times the baseline rate, plus 15 percentage points (10 for themes). Alerts are labelled as **detected patterns, not proof of a cause**.

### Fake-review risk
> **This is a risk indicator, not proof that a review is fake.** Genuine customers can trigger these signals. Use the score to decide which reviews deserve a closer look, never to accuse a reviewer.

| Signal | Points |
| --- | --- |
| Near-duplicate text (similarity >= 0.90 to another review) | +40 |
| Very similar text (similarity >= 0.75) | +20 |
| Review burst: 3+ same-rating reviews of one product on one day, well above its normal daily volume | +20 |
| Generic text: 6 words or fewer, with no specific product detail | +15 |
| Promotional phrasing ("must buy", "best ever", "highly recommended", ...) | +10 |
| Rating contradicts text: 4–5 stars with clearly negative text, or 1–2 stars with clearly positive text (mixed reviews excluded) | +15 |
| Shouting ("!!!" or mostly capital letters) | +10 |
| Extreme rating (1 or 5 stars), added only when another signal is present | +5 |

Similarity is calculated with TF-IDF over words and counts only between reviews of similar length, so a short review contained in a longer one is not treated as a copy. Levels: **High >= 55, Medium >= 30, Low < 30**. A single weak signal never reaches Medium. Each flagged review lists its reasons.

### Health score
The health score is a 0–100 summary: half comes from the average rating (scaled from 1–5) and half from net sentiment (positive share minus negative share).

---

## Testing

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

The tests never call the real AI API. They cover:
- **Data:** cleaning and validation, column aliases, and that loading never creates files.
- **Sentiment and language:** VADER labels, the review vocabulary, and Hinglish/Hindi sentiment and language detection (including the three Hinglish examples from the project brief).
- **Themes:** whole-word matching, praise and complaint in the same review, and "sound quality" vs Product Quality.
- **Priority and insights:** ranking and levels, and that evidence IDs exist in the data.
- **Trends and alerts:** trend shapes, no alerts on small datasets, and the Connectivity spike in the demo data.
- **Fake-review risk:** the burst of near-identical reviews is flagged High, and most reviews stay Low.
- **Ask Your Reviews:** retrieval finds the relevant reviews.
- **AI service (with a mocked model):** prompts include the statistics and the question, cited IDs are checked, large datasets are batched, and the missing-key, empty-question and network-error messages are friendly.

To regenerate the synthetic demo data:

```powershell
python scripts/generate_demo_data.py
```

---

## Deployment (Streamlit Community Cloud)

1. Push the project to a GitHub repository. `.env` and `.streamlit/secrets.toml` are git-ignored, so keys stay local.
2. On https://share.streamlit.io, create a new app from the repository with `app.py` as the main file.
3. In the app's **Settings → Secrets**, add:

   ```toml
   AI_API_KEY = "your_api_key_here"
   AI_PROVIDER = "gemini"
   ```

   Streamlit Community Cloud exposes top-level secrets as environment variables, and `ai_service.py` reads environment variables, so no code changes are needed.
4. Deploy. The app works without the secret too, with AI sections disabled.

---

## Limitations

- **AI output can still be wrong.** Check it against the cited evidence. The quality of AI output depends on the model chosen.
- **Hindi/Hinglish accuracy is limited.** The lexicon covers common words only. Romanised spelling varies a lot, and sarcasm and complex grammar are often missed. Accuracy is lower than for English.
- **VADER and the keyword themes are rule-based.** They can miss unusual phrasing or product-specific vocabulary. New themes and keywords can be added in `themes.py`.
- **Fake-review risk is a heuristic.** It can't prove anything. Genuine short or enthusiastic reviews may score Medium.
- **Spike alerts need enough data**, and they show correlation, not cause.
- **The bundled datasets are synthetic** and simpler than real review data.

## Future improvements

- A multilingual transformer sentiment model, so Hindi/Hinglish accuracy doesn't depend on a word list.
- Learned topics (e.g. embeddings with clustering) alongside the keyword themes.
- Reviewer-level signals for fake-review risk, such as account age and review history, when the data includes them.
- Scheduled data imports from marketplaces, and alert notifications by email or Slack.
- Comparing products side by side.

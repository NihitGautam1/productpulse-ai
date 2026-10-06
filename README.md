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
| 14 | Video reviews (YouTube links and uploaded video files) | IMPLEMENTED | `video_reviews.py`: YouTube captions, or Gemini transcription |
| 15 | Video moment links: jump to the second a reviewer praises or criticises something | IMPLEMENTED | Caption timestamps (`video_reviews.py`) |
| 16 | Head-to-head product comparison with an AI verdict | IMPLEMENTED | `compare.py`, `ai_service.py` |
| 17 | Dashboard headline (computed, or written by AI) with an animated health ring | IMPLEMENTED | `insights.py`, `styles.py` |
| 18 | Theme map, review cards and video cards | IMPLEMENTED | `charts.py`, `styles.py` |
| 19 | Emotion detection (anger, frustration, disappointment, confusion, delight, satisfaction) | IMPLEMENTED | Keyword lexicon with negation handling (`emotions.py`) |
| 20 | Feature wishlist: what customers ask you to build, grouped and ranked | IMPLEMENTED | Request patterns + TF-IDF grouping (`wishlist.py`), optional AI summary |
| 21 | Reply Studio: draft replies to reviews in three tones | IMPLEMENTED | Templates (`replies.py`), or AI-written per review |
| 22 | "If you fixed it" rating estimate per issue | IMPLEMENTED | `insights.fix_impact` |
| 23 | Downloadable PDF report | IMPLEMENTED | `report.py` (fpdf2) |
| 24 | Word clouds of praise and complaint words | IMPLEMENTED | `insights.top_words` |
| 25 | Saved video reviews, YouTube view counts and audience reach | IMPLEMENTED | `video_reviews.py` |
| 26 | Welcome tour | IMPLEMENTED | `views.welcome` |
| 27 | Upload several CSV files at once, with automatic column detection and manual mapping | IMPLEMENTED | `csv_normalizer.py`, `upload_ui.py` |

All AI features are optional. **AI output quality depends on the model. The AI features were verified with mocked responses in the automated tests.** Everything that doesn't use AI works with no API key.

---

## Dashboard pages

| Page | What it shows |
| --- | --- |
| **Overview** | A headline banner with an animated health ring (the headline can be rewritten by AI), a theme map bubble chart, key numbers (reviews, average rating, positive/negative share, health score) with mini trend charts, active alerts, sentiment mix, rating distribution, top priority issues with an example quote, most-praised features, key insights, and the AI summary |
| **Compare Products** | Two products side by side: a "VS" panel with winners marked, a theme satisfaction radar, star ratings, clear differences and an AI verdict with cited reviews |
| **AI Insights** | AI overall summary, positive feedback, negative feedback, key insights and recommended actions, each with a panel listing the cited reviews. Data-driven insights are always shown, even without AI |
| **Ask Your Reviews** | Ask questions in plain language. Answers come from the most relevant reviews and computed statistics, with the cited reviews shown |
| **Priority Issues** | "If you fixed it": the estimated rating gain for each issue, plus ranked issues with their score breakdown, praise vs complaints per theme, and real review quotes as evidence for each issue |
| **Feature Wishlist** | Feature requests ("I wish it had…", "please add…", "…hona chahiye") grouped and ranked, with an optional AI wishlist and quick wins |
| **Reply Studio** | Negative reviews with a "Draft a reply" button: friendly, professional or apologetic, with your signature. Template-based without AI |
| **Trends & Alerts** | Review volume, negative share, average rating and complaints per theme over time, plus spike alerts |
| **Review Integrity** | Fake-review risk levels with the reasons for each flag, and a clear disclaimer |
| **Review Explorer** | All reviews as cards (praise highlighted blue, complaints red) or as a table, with search and filters (sentiment, language, risk, rating, theme, source), a CSV download of the enriched data, and the language mix |
| **Video Reviews** | Add review videos by YouTube link or file upload. Each video is transcribed and joins the analysis on every page as one review. Videos show as thumbnail cards with key moments linked to the exact time in the video |

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
| `ai_service.py` | All AI/API code: prompts, grounding rules, chunking, citation checking, video transcription, friendly errors |
| `video_reviews.py` | Turns YouTube links and uploaded videos into review rows (transcript as the review text); moment links |
| `compare.py` | Head-to-head product metrics and per-theme satisfaction |
| `emotions.py` | Emotion per review (English, Hinglish, Hindi keywords; ignores negated positives like "not good") |
| `wishlist.py` | Finds feature requests and groups similar ones |
| `replies.py` | Template replies to reviews (the AI version is in `ai_service.py`) |
| `report.py` | Builds the PDF report |
| `csv_normalizer.py` | Reads uploaded CSVs, detects columns, converts them to the standard schema, cleans and combines files |
| `upload_ui.py` | The multi-file upload interface: summary, mapping forms, duplicate and rating-scale choices |
| `Start ProductPulse.bat` | Double-click to start the app (or just open it if it is already running) |
| `packages.txt` | Fonts installed on Streamlit Community Cloud (for the PDF report, including Hindi) |
| `scripts/generate_demo_data.py` | Rebuilds the synthetic demo dataset (fixed random seed) |
| `tests/` | Automated tests (pytest). They never call the real API |
| `.streamlit/config.toml` | Theme (light and dark), fonts, upload limit |
| `.env.example` | Template for your local `.env` (placeholders only) |

---

## Video reviews

Open **Video Reviews** (under Trust & data) to add videos. Each video becomes one review whose text is the transcript, so sentiment, themes, priority, trends, fake-review signals and the AI features all include it.

- **YouTube links:** paste one or more links (watch, youtu.be, Shorts or embed). The video's captions are used when it has them, which needs no API key. Videos without captions are transcribed by Gemini from the link (public videos only).
- **Video files:** MP4, MOV, WebM, MKV, AVI and other common formats, plus audio files (MP3, WAV, M4A), up to 500 MB each. The file is sent to Gemini for transcription and deleted from Gemini afterwards. This needs `AI_PROVIDER=gemini`; Claude cannot transcribe audio.
- **Product, rating and date:** choose the product the video reviews (YouTube videos default to the video title). Videos have no star rating, so unless you enter one it is estimated from the sentiment of the transcript and marked "estimated". The date defaults to the YouTube upload date, or today for uploaded files.
- Added videos are saved to `.productpulse/video_reviews.json` (ignored by git), so they come back the next time the app starts. The page shows total YouTube views and the views of negative videos (their reach).
- **Key moments:** for YouTube videos with captions, every praise or complaint found in the transcript links to the second it is said (`watch?v=...&t=277s`). Evidence quotes and review cards elsewhere in the app carry the same "▶ Watch at" link. Videos transcribed by Gemini have no timestamps, so they have no moment links.
- Video reviews appear as ★ markers on the Trends average-rating chart.
- Old videos default to their YouTube upload date, which can stretch the date range. Set the review date when adding them if you want them inside your current period.
- In AI prompts a transcript can use up to 8,000 characters (written reviews: 1,000) and is labelled as a video transcript.

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

**Quick start after installation:** double-click **`Start ProductPulse.bat`** (or the *ProductPulse AI* shortcut on the desktop). It starts the app and opens http://localhost:8501, which is always the same link. If the app is already running it just opens the browser. Keep the window it opens while you use the app; closing it stops the app.

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

## Dataset format and CSV upload

### Standard format

```text
review_id,product_name,rating,review_text,review_date
R001,Nimbus Headphones,5,Noise cancellation is outstanding.,2026-02-10
```

Inside the app everything uses these five columns. Uploaded files are converted to them first (`csv_normalizer.py`), and the rest of the app is unchanged:

```text
uploaded CSV files -> column detection / mapping -> review_id, product_name, rating, review_text, review_date -> existing analysis
```

### Uploading several files at once

1. In the sidebar, choose **Dataset → Upload my own CSV files**.
2. Under **Upload Review CSV Files**, select one or more CSV files (or drag them in).
3. Each file is read and its columns are detected separately, then all usable files are combined into one dataset. You never need to merge files yourself.
4. An **Upload summary** appears at the top of the page. It lists each file (✅ processed, 🧩 needs column mapping, ❌ could not be read, with the reason), the total number of reviews, the number of products, how each column was mapped, and any warnings.

A file that cannot be used never blocks the others: valid files are analysed straight away. The **Product** filter is always built from the combined `product_name` values, so it changes whenever the uploaded files change. The Review Explorer gets a **File** filter when reviews come from more than one file.

### Supported column names

Matching ignores case, spaces, underscores and hyphens, so `Review ID`, `review_id`, `ReviewID` and `review-id` are all the same.

| Standard column | Recognised names (examples) |
| --- | --- |
| `review_id` | review_id, reviewid, review id, id, review_number, review_no, review_number_id, comment_id, uid |
| `product_name` | product_name, product, product name, product_title, item, item_name, model, name. If none exists: product_id, asin, sku |
| `rating` | rating, ratings, stars, star_rating, score, review_rating, review score, overall, rate |
| `review_text` | review_text, review, review text, comment, comments, feedback, feedback_text, text, content, description, review_body, body, message |
| `review_date` | review_date, date, review_time, timestamp, created_at, created_date, posted_date, published_date, time, datetime |

Example formats that work without any setup:

```text
review_id,product_name,rating,review_text,review_date        (standard)
id,product,review,stars,date                                 (alternative)
Review ID,Product Name,Rating,Review Text,Review Date        (different capitalisation)
reviewID,product,feedback,score,created_at                   (mixed names, Unix timestamps, 1-10 scores)
```

### Automatic column detection

Detection goes from most to least certain, and every match is checked against the values in the column:

1. **Known names** (the table above). An ambiguous name such as `review` is used for whichever field its values fit (long text → review text, short codes → review ID).
2. **Similar names**, such as `Review Rating (out of 5)` or `reviewText`.
3. **The values alone**, only when exactly one remaining column clearly fits: ratings look like `5`, `4.5`, `5 stars`, `4/5` or `★★★★`; dates parse as dates (or are Unix timestamps); review text is long free text. Product names and IDs are never guessed from values alone.

A column whose name matches but whose values don't fit (for example a `score` column full of words) is not used. If a review ID column is missing, IDs are created from the file name (`reviews-1`, `reviews-2`, ...). Anything else that can't be identified confidently is left for you to map.

### Manual column mapping

When a required column can't be identified, the file shows **🧩 Column mapping needed** with a preview of the file and five dropdowns (Review ID, Product Name, Rating, Review Text, Review Date). Choose the column for each and press **Apply Mapping**. If the file really has no such column you can choose:

- Review ID: *create IDs automatically*
- Product Name: *use the file name as the product*
- Rating: *estimate stars from the review text* (sentiment-based, shown as estimated)
- Review Date: *use today's date*

Review Text must always be a real column. Automatically mapped files can be adjusted with **Change mapping** in the upload summary.

### Duplicate review IDs

When files are combined, duplicates are reported in the upload summary:

- **Exact duplicates** (all five fields identical) are kept by default. Choose **Remove exact duplicates** to drop them.
- **Same ID, different content** is never deleted. Both reviews are kept and the later ones get a `~2`, `~3` suffix (for example `R001~2`) so that evidence and AI citations always point to exactly one review. A warning says how many IDs were affected.

### Data cleaning rules

1. Completely empty rows are removed.
2. Rows without review text are removed (with a count in the warnings).
3. Ratings are converted to numbers. `5`, `"5"`, `"5 stars"`, `"4.5"`, `"3,5"`, `"★★★"` are read as written; fractions such as `"4/5"` or `"8 out of 10"` are converted to a 1-5 value.
4. If a file's ratings are on another scale (1-10, 0-10 or 0-100), the summary asks whether to convert them to 1-5 (converted by default). Ratings outside 1-5 after that are removed with a warning.
5. Dates are parsed in common formats, ISO timestamps with time zones, and Unix seconds or milliseconds. Rows whose date is missing or unreadable are left out, and the summary says how many. (Ambiguous dates such as `03/04/2026` are read month-first unless the day is above 12.)
6. Leading and trailing spaces are trimmed; the review text itself is kept as written.
7. Files are read as UTF-8, UTF-8 with BOM, Windows-1252 or Latin-1, and comma, semicolon or tab separators are detected automatically. Hindi and Hinglish text is supported.

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
- **CSV upload (`tests/test_csv_normalizer.py`):** the standard format, the `id, product, review, stars, date` format, different capitalisation, several files with different schemas, a valid file plus an invalid one, duplicate IDs (exact and conflicting), missing dates, missing review text, ratings stored as strings and on a 1-10 scale, empty files, a 50,000-row file, Hindi/Hinglish text, Windows-encoded and semicolon-separated files, detection by values, manual mapping, and the combined upload running through the full analysis.
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

   `app.py` copies the `AI_*` secrets into the environment at start-up, where `ai_service.py` reads them.
4. Deploy. The app works without the secret too, with AI sections disabled.

On the cloud, saved video reviews (`.productpulse/`) last only until the app restarts, and free apps go to sleep after a few days without visitors (they wake up when opened, in about 30 seconds). `packages.txt` installs the fonts the PDF report needs there.

---

## Limitations

- **AI output can still be wrong.** Check it against the cited evidence. The quality of AI output depends on the model chosen.
- **Hindi/Hinglish accuracy is limited.** The lexicon covers common words only. Romanised spelling varies a lot, and sarcasm and complex grammar are often missed. Accuracy is lower than for English.
- **VADER and the keyword themes are rule-based.** They can miss unusual phrasing or product-specific vocabulary. New themes and keywords can be added in `themes.py`.
- **Fake-review risk is a heuristic.** It can't prove anything. Genuine short or enthusiastic reviews may score Medium.
- **Spike alerts need enough data**, and they show correlation, not cause.
- **The bundled datasets are synthetic** and simpler than real review data.
- **CSV column detection is rule-based.** Unusual column names or values may still need manual mapping. Dates like `03/04/2026` are read month-first. A file with ratings on an unusual scale (above 100) is not converted.
- **Very large uploads** (tens of thousands of reviews) are read quickly, but the analysis and AI calls take longer; AI features use the most recent 1,500 reviews.

## Future improvements

- A multilingual transformer sentiment model, so Hindi/Hinglish accuracy doesn't depend on a word list.
- Learned topics (e.g. embeddings with clustering) alongside the keyword themes.
- Reviewer-level signals for fake-review risk, such as account age and review history, when the data includes them.
- Scheduled data imports from marketplaces, and alert notifications by email or Slack.

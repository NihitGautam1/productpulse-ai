# ProductPulse AI

A beginner-friendly **customer review intelligence dashboard** for sellers. It reads a CSV of product reviews, scores each review with **VADER** sentiment (runs on your computer, no API keys), then shows ratings, sentiment mix, top complaints, and a searchable table.

All dashboard numbers are **calculated from the dataset**. If the CSV is missing or invalid, the app builds a small **synthetic sample** and labels it clearly.

## What each file does

| File | Purpose in simple language |
| --- | --- |
| `app.py` | The Streamlit website you open in the browser. It shows the overview, charts, and table. |
| `data_loader.py` | Loads `data/reviews.csv`, checks columns, cleans bad rows, and creates sample data if the file is missing or broken. |
| `sentiment.py` | Reads each review and labels it Positive, Negative, or Neutral using VADER. |
| `themes.py` | Looks for common complaint topics (shipping, quality, service, price, comfort) in unhappy reviews. |
| `data/reviews.csv` | The reviews file. Columns: `review_id`, `product_name`, `rating`, `review_text`, `review_date`. |
| `requirements.txt` | The Python packages you need to install. |
| `README.md` | This guide. |

## CSV columns

```text
review_id,product_name,rating,review_text,review_date
```

- `rating` should be a number from 1 to 5.
- `review_date` should be a date such as `2026-03-15`.

## Windows setup and run commands

Open **PowerShell**. Run these commands **exactly**.

```powershell
cd C:\Users\Hp\productpulse-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import pandas, matplotlib, streamlit; from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; print('imports ok')"
streamlit run app.py
```

If PowerShell blocks the virtual environment script, run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

The app opens at **http://localhost:8501**.

To stop the server, press `Ctrl+C` in PowerShell.

To use your own reviews, replace `data\reviews.csv` (keep the same column names) and refresh the browser.

## How metrics are calculated

- **Total reviews**: number of valid rows after cleaning.
- **Average rating**: mean of the `rating` column.
- **Sentiment chart**: VADER compound score per review (`>= 0.05` positive, `<= -0.05` negative, otherwise neutral).
- **Top complaints**: keyword themes counted only in negative sentiment or 1–2 star reviews.

## Requirements

- Windows 10/11
- Python 3.10 or newer
- CPU only; internet is needed only to install packages the first time

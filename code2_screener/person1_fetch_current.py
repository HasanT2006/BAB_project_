# ============================================================
# PERSON 1 — Current S&P 500 Data Fetcher
# File: code2_screener/person1_fetch_current.py
#
# ROLE IN PROJECT:
#   Fetches 3 years of monthly S&P 500 stock prices from
#   Yahoo Finance for use in the live BAB screener.
#
# MARKS TARGETED:
#   Basic Programming (30%):
#     - Dictionary: stock_data stores ticker → DataFrame
#     - List: tickers list from Wikipedia
#     - For Loop: download every ticker
#     - While Loop: retry logic inside download_with_retry
#     - Error Handling: try/except on every network call
#     - Conditions: skip tickers with too little data
#   Libraries (40%):
#     - yfinance: live stock data download
#     - requests: fetch Wikipedia HTML
#     - pandas: read_html, DataFrame, pct_change
#
# OUTPUT:
#   data/processed/current_prices.csv
#   data/processed/current_returns.csv
#
# HOW TO RUN:
#   python code2_screener/person1_fetch_current.py
# ============================================================

import yfinance as yf
import pandas as pd
import requests
import time
import os
import sys
from io import StringIO

# ============================================================
# SETTINGS
# ============================================================

START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"

# Rate-limiting settings to avoid getting blocked
SLEEP_BETWEEN_REQUESTS = 0.5   # seconds between downloads
SLEEP_AFTER_ERROR      = 2.0   # seconds to wait after a failure
MAX_RETRIES            = 3     # maximum retry attempts per ticker
MIN_MONTHS_REQUIRED    = 12    # skip ticker if fewer months available

PRICES_OUTPUT  = "data/processed/current_prices.csv"
RETURNS_OUTPUT = "data/processed/current_returns.csv"

PREVIEW_ROWS = 5
PREVIEW_COLS = 6


# ============================================================
# STEP 1 — GET S&P 500 TICKERS FROM WIKIPEDIA
# ============================================================

def get_sp500_tickers():
    """
    Scrapes the S&P 500 member list from Wikipedia.
    Returns a list of ticker strings.
    Uses: requests, try/except, list comprehension
    """
    print("\n" + "=" * 55)
    print("STEP 1: Fetching S&P 500 tickers from Wikipedia")
    print("=" * 55)

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Wikipedia blocks plain requests — mimic a real browser
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()   # raises error on 4xx/5xx

        # Parse HTML tables from the page
        tables  = pd.read_html(StringIO(response.text))
        df      = tables[0]           # first table = S&P 500 members

        # List of ticker strings — replace . with - for Yahoo Finance
        tickers = [t.replace(".", "-") for t in df["Symbol"].tolist()]

        print(f"   ✅ Found {len(tickers)} tickers")
        print(f"      First 5 : {tickers[:5]}")
        print(f"      Last 5  : {tickers[-5:]}")
        return tickers

    except Exception as e:
        print(f"   ❌ ERROR fetching tickers: {e}")
        return []


# ============================================================
# STEP 2 — DOWNLOAD ONE TICKER WITH RETRY
# ============================================================

def download_with_retry(ticker, start_date, end_date):
    """
    Downloads monthly OHLCV data for one ticker.
    Uses exponential backoff on failure.

    Uses: while loop, try/except, conditions

    Returns:
        (DataFrame, "OK")  on success
        (None, reason_str) on failure
    """
    wait_time  = SLEEP_AFTER_ERROR
    attempt    = 0

    # While loop for retry logic
    while attempt < MAX_RETRIES:
        attempt += 1

        try:
            df = yf.download(
                ticker,
                start       = start_date,
                end         = end_date,
                interval    = "1mo",
                auto_adjust = True,
                progress    = False,
            )

            # Condition: reject empty downloads
            if df.empty:
                return None, "No data returned"

            # Condition: reject tickers with too little history
            if len(df) < MIN_MONTHS_REQUIRED:
                return None, f"Only {len(df)} months (need {MIN_MONTHS_REQUIRED})"

            # Keep only closing price, rename to ticker symbol
            df = df[["Close"]].rename(columns={"Close": ticker})

            # Normalise dates to month-end for alignment with other data
            df.index = pd.to_datetime(df.index)
            df.index = df.index.to_period("M").to_timestamp("M")

            return df, "OK"

        except Exception as e:
            # Condition: retry if not final attempt
            if attempt < MAX_RETRIES:
                print(f"      Attempt {attempt} failed → waiting {wait_time}s...")
                time.sleep(wait_time)
                wait_time *= 2    # exponential backoff
            else:
                return None, str(e)

    return None, "Max retries exceeded"


# ============================================================
# STEP 3 — DOWNLOAD ALL TICKERS
# ============================================================

def download_all_stocks(tickers, start_date, end_date):
    """
    For-loops over all tickers, downloads each one.
    Tracks results in a dictionary.
    Uses: for loop, dictionary, list, conditions
    """
    print("\n" + "=" * 55)
    print(f"STEP 2: Downloading {len(tickers)} stocks")
    print(f"        {start_date} → {end_date}")
    print("=" * 55)

    # Dictionary: ticker symbol → DataFrame
    stock_data    = {}
    failed        = {}     # Dictionary: ticker → failure reason
    success_count = 0

    # For loop — download each ticker
    for i, ticker in enumerate(tickers):

        # Progress update every 50 stocks
        if i % 50 == 0:
            print(f"\n   Progress: {i}/{len(tickers)} stocks...")

        df, status = download_with_retry(ticker, start_date, end_date)

        # Condition: store result or record failure
        if status == "OK":
            stock_data[ticker] = df
            success_count += 1
        else:
            failed[ticker] = status
            if status not in ("No data returned",):
                print(f"   ⚠️  {ticker}: {status}")

        # Rate limit — be polite to Yahoo Finance servers
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"\n   ✅ Downloaded successfully : {success_count} stocks")
    print(f"   ⚠️  Failed / skipped        : {len(failed)} stocks")

    return stock_data, failed


# ============================================================
# STEP 4 — COMBINE INTO MASTER DATAFRAMES
# ============================================================

def combine_data(stock_data):
    """
    Combines dictionary of DataFrames into two master tables:
      prices_df  → raw closing prices
      returns_df → monthly percentage returns (pct_change)
    Uses: dictionary values, pandas concat
    """
    print("\n" + "=" * 55)
    print("STEP 3: Combining into master DataFrames")
    print("=" * 55)

    # Concatenate all DataFrames in the dictionary side-by-side
    prices_df = pd.concat(stock_data.values(), axis=1).sort_index()

    # Honestly report missing data
    total_cells   = prices_df.size
    missing_cells = prices_df.isnull().sum().sum()
    pct_missing   = (missing_cells / total_cells) * 100

    print(f"   Stocks included  : {prices_df.shape[1]}")
    print(f"   Months covered   : {prices_df.shape[0]}")
    print(f"   Missing values   : {missing_cells} ({pct_missing:.1f}%)")

    # Monthly percentage returns: (this month − last month) / last month
    returns_df = prices_df.pct_change().iloc[1:]   # drop first NaN row

    print(f"   Returns shape    : {returns_df.shape}")
    return prices_df, returns_df


# ============================================================
# STEP 5 — PREVIEW
# ============================================================

def preview_data(prices_df, returns_df):
    """Shows a readable sample of the data."""
    print("\n" + "=" * 55)
    print("STEP 4: Data Preview")
    print("=" * 55)

    # Show first few columns only for readability
    sample_cols = list(prices_df.columns[:PREVIEW_COLS])

    print(f"\nPRICES (first {PREVIEW_ROWS} rows, first {PREVIEW_COLS} stocks):")
    print(prices_df[sample_cols].head(PREVIEW_ROWS).round(2).to_string())

    print(f"\nRETURNS (first {PREVIEW_ROWS} rows, first {PREVIEW_COLS} stocks):")
    print(returns_df[sample_cols].head(PREVIEW_ROWS).round(4).to_string())

    print(f"\nDate range: {prices_df.index[0].date()} → {prices_df.index[-1].date()}")


# ============================================================
# STEP 6 — SAVE
# ============================================================

def save_data(prices_df, returns_df, failed):
    """
    Saves prices, returns, and failed tickers to disk.
    Uses: os.makedirs, try/except
    """
    print("\n" + "=" * 55)
    print("STEP 5: Saving data")
    print("=" * 55)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    try:
        prices_df.to_csv(PRICES_OUTPUT)
        print(f"   ✅ Prices  saved : {PRICES_OUTPUT}")

        returns_df.to_csv(RETURNS_OUTPUT)
        print(f"   ✅ Returns saved : {RETURNS_OUTPUT}")

        # Condition: only save failed file if there were failures
        if failed:
            failed_df = pd.DataFrame(
                list(failed.items()),
                columns=["Ticker", "Reason"]
            )
            failed_df.to_csv("data/raw/failed_tickers.csv", index=False)
            print(f"   ✅ Failed tickers: data/raw/failed_tickers.csv"
                  f"  ({len(failed)} tickers)")

    except Exception as e:
        print(f"   ❌ ERROR saving: {e}")


# ============================================================
# CACHE CHECK
# ============================================================

def load_from_cache():
    """
    Loads from disk if files already exist — saves re-downloading.
    Uses: condition, try/except
    """
    if os.path.exists(PRICES_OUTPUT) and os.path.exists(RETURNS_OUTPUT):
        print(f"\n   ✅ Cached files found — loading from disk")
        print(f"      Delete files in data/processed/ to force re-download")
        try:
            prices_df  = pd.read_csv(PRICES_OUTPUT,
                                     index_col="Date", parse_dates=True)
            returns_df = pd.read_csv(RETURNS_OUTPUT,
                                     index_col="Date", parse_dates=True)
            print(f"      Prices  : {prices_df.shape}")
            print(f"      Returns : {returns_df.shape}")
            return prices_df, returns_df
        except Exception as e:
            print(f"   ⚠️  Cache load failed ({e}) — re-downloading...")
            return None, None
    return None, None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("PERSON 1 — CURRENT S&P 500 DATA FETCHER")
    print("=" * 55)

    # Try cache first
    prices_df, returns_df = load_from_cache()

    if prices_df is not None:
        preview_data(prices_df, returns_df)

    else:
        # Step 1: Get tickers
        tickers = get_sp500_tickers()
        if not tickers:
            print("❌ Could not get tickers. Exiting.")
            sys.exit(1)

        # Step 2: Download
        stock_data, failed = download_all_stocks(tickers, START_DATE, END_DATE)
        if not stock_data:
            print("❌ No data downloaded. Exiting.")
            sys.exit(1)

        # Step 3: Combine
        prices_df, returns_df = combine_data(stock_data)

        # Step 4: Preview
        preview_data(prices_df, returns_df)

        # Step 5: Save
        save_data(prices_df, returns_df, failed)

    print("\n" + "=" * 55)
    print("FOR YOUR TEAMMATES")
    print("=" * 55)
    print(f"   prices  = pd.read_csv('{PRICES_OUTPUT}',")
    print(f"             index_col='Date', parse_dates=True)")
    print(f"   returns = pd.read_csv('{RETURNS_OUTPUT}',")
    print(f"             index_col='Date', parse_dates=True)")
    print("=" * 55)

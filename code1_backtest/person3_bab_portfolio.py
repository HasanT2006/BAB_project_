# ============================================================
# PERSON 3 — BAB Portfolio Construction & Backtesting
# File: code1_backtest/person3_bab_portfolio.py
#
# ROLE IN PROJECT:
#   Takes Person 2's rolling betas and builds the actual
#   Betting-Against-Beta portfolio month by month.
#   Implements the Frazzini & Pedersen (2014) construction:
#     - Long the low-beta half, short the high-beta half
#     - Weights proportional to distance from median beta
#     - Each leg scaled so its beta = 1 → net beta ≈ 0
#   Produces the equity curve and performance metrics that
#   Person 4 will test and Person 5 will visualise.
#
# MARKS TARGETED:
#   Basic Programming (30%):
#     - For Loop: month-by-month portfolio construction
#     - Dictionary: stores monthly returns and weights
#     - Functions: each stage is its own function
#     - Conditions: skip months with insufficient data
#     - Error Handling: try/except on file loading
#     - List: used for collecting equity curve values
#   Libraries (40%):
#     - pandas: Series, DataFrame, reindex, concat
#     - numpy: sqrt for annualisation, cumprod for equity curve
#
# INPUTS:
#   data/processed/current_returns.csv   (Person 1)
#   data/processed/aqr_returns.csv       (Person 1, optional)
#
# OUTPUTS:
#   data/processed/bab_portfolio_returns.csv
#   data/processed/bab_portfolio_weights.csv
#   data/processed/bab_equity_curve.csv
#
# HOW TO RUN:
#   python code1_backtest/person3_bab_portfolio.py
# ============================================================

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# ── Import Person 2's beta functions ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from person2_beta_calculation import (
    calculate_rolling_beta,
    split_into_high_low_beta,
)

# ============================================================
# SETTINGS
# ============================================================

RETURNS_PATH  = "data/processed/current_returns.csv"
AQR_PATH      = "data/processed/aqr_returns.csv"

PORTFOLIO_OUT = "data/processed/bab_portfolio_returns.csv"
WEIGHTS_OUT   = "data/processed/bab_portfolio_weights.csv"
EQUITY_OUT    = "data/processed/bab_equity_curve.csv"

BETA_WINDOW        = 12     # rolling window in months
MIN_PERIODS        = 6      # minimum data points needed in window
HIGH_FRACTION      = 0.50   # top 50% → short leg
LOW_FRACTION       = 0.50   # bottom 50% → long leg
MIN_STOCKS_PER_LEG = 5      # skip month if fewer stocks in either leg

PREVIEW_ROWS = 6


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

def load_data() -> tuple[DataFrame, Series, Series]:
    """
    Loads stock returns, market returns, and risk-free rate.
    Uses: try/except, conditions, dictionary-style column access

    Returns:
        stock_returns : DataFrame (date × tickers)
        market_ret    : Series   (date,)
        rf            : Series   (date,)
    """
    print("\n" + "=" * 55)
    print("STEP 1: Loading data")
    print("=" * 55)

    # Error handling: check file exists before loading
    if not os.path.exists(RETURNS_PATH):
        raise FileNotFoundError(
            f"Stock returns not found at '{RETURNS_PATH}'.\n"
            "Run person1_fetch_current.py first."
        )

    try:
        stock_returns = pd.read_csv(
            RETURNS_PATH, index_col="Date", parse_dates=True
        )
        print(f"   ✅ Stock returns : {stock_returns.shape[0]} months × "
              f"{stock_returns.shape[1]} stocks")
    except Exception as e:
        raise RuntimeError(f"Failed to load stock returns: {e}")

    # Load AQR factors — optional, fall back gracefully if missing
    market_ret = pd.Series(dtype=float)
    rf         = pd.Series(dtype=float)

    if os.path.exists(AQR_PATH):
        try:
            aqr = pd.read_csv(AQR_PATH, index_col="Date", parse_dates=True)

            # Condition: only use column if it exists
            if "MKT" in aqr.columns:
                market_ret = aqr["MKT"]
                print(f"   ✅ Market returns : {len(market_ret)} months")
            if "RF" in aqr.columns:
                rf = aqr["RF"]
                print(f"   ✅ Risk-free rate : {len(rf)} months")
        except Exception as e:
            print(f"   ⚠️  Could not load AQR data: {e}")
    else:
        print(f"   ⚠️  AQR file not found — using equal-weighted proxy")

    return stock_returns, market_ret, rf


# ============================================================
# STEP 2 — COMPUTE ROLLING BETAS
# ============================================================

def compute_betas(
    stock_returns : DataFrame,
    market_ret    : Series,
    rf            : Series,
) -> DataFrame:
    """
    Delegates beta computation to Person 2's module.
    Falls back to equal-weighted market if no market series available.
    Uses: condition
    """
    print("\n" + "=" * 55)
    print("STEP 2: Computing rolling betas (calls Person 2 module)")
    print("=" * 55)

    # Condition: build proxy market if AQR not available
    use_rf = len(rf) > 0
    if len(market_ret) == 0:
        print("   ⚠️  No market series — building equal-weighted proxy")
        market_ret = stock_returns.mean(axis=1)
        market_ret.name = "EW_Market"
        use_rf = False

    beta_df = calculate_rolling_beta(
        asset_returns  = stock_returns,
        market_returns = market_ret,
        window         = BETA_WINDOW,
        min_periods    = MIN_PERIODS,
        rf             = rf if use_rf else None,
        adjust_for_rf  = use_rf,
    )

    # Count months with at least one valid beta
    valid_months = int(beta_df.notna().any(axis=1).sum())
    print(f"   ✅ Beta matrix   : {beta_df.shape[0]} months × "
          f"{beta_df.shape[1]} stocks")
    print(f"      Valid months  : {valid_months}")

    return beta_df


# ============================================================
# STEP 3 — COMPUTE PORTFOLIO WEIGHTS (F&P 2014)
# ============================================================

def compute_bab_weights(
    beta_row : Series,
) -> tuple[Series, Series]:
    """
    Computes long and short weights for ONE month.

    Frazzini & Pedersen (2014) Algorithm:
      1. Split universe into bottom 50% (long) and top 50% (short).
      2. Raw weight = |beta − cross-sectional median beta|.
         Stocks further from the median get larger positions.
      3. Normalise long and short legs separately so each sums to 1.
      4. Short leg gets negative sign (short positions).

    Parameters:
        beta_row : Series of betas for all stocks in one month.

    Returns:
        long_w  : positive weights for low-beta stocks
        short_w : negative weights for high-beta stocks
    """
    # Drop NaN tickers for this month
    valid = beta_row.dropna()

    # Condition: skip month if not enough stocks
    if len(valid) < MIN_STOCKS_PER_LEG * 2:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    median_beta = valid.median()

    # Quantile thresholds
    low_cut  = valid.quantile(LOW_FRACTION)
    high_cut = valid.quantile(1.0 - HIGH_FRACTION)

    # Boolean masks for each leg
    low_mask  = valid <= low_cut
    high_mask = valid >= high_cut

    # Condition: need enough stocks in each leg
    if low_mask.sum() < MIN_STOCKS_PER_LEG or \
       high_mask.sum() < MIN_STOCKS_PER_LEG:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Raw weights = distance from median
    low_raw  = (median_beta - valid[low_mask]).abs()
    high_raw = (valid[high_mask] - median_beta).abs()

    # Condition: avoid division by zero
    if low_raw.sum() == 0 or high_raw.sum() == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Normalise: each leg sums to 1
    long_w  =  low_raw  / low_raw.sum()    # positive weights
    short_w = -high_raw / high_raw.sum()   # negative weights

    return long_w, short_w


# ============================================================
# STEP 4 — BUILD PORTFOLIO MONTH BY MONTH
# ============================================================

def build_portfolio(
    beta_df       : DataFrame,
    stock_returns : DataFrame,
) -> tuple[Series, DataFrame]:
    """
    Constructs the BAB portfolio using a for loop over months.

    Signal timing:
        Weights from month t → returns realised in month t+1
        (avoids look-ahead bias)

    Beta scaling (F&P 2014):
        Each leg is further scaled so its weighted-average beta = 1.
        This ensures the combined portfolio has net beta ≈ 0.

    Uses: for loop, dictionary (port_returns, weight_store),
          conditions, list operations

    Returns:
        port_returns : Series — monthly BAB portfolio returns
        weight_df    : DataFrame — signed weights for every month
    """
    print("\n" + "=" * 55)
    print("STEP 3: Building BAB portfolio month by month")
    print("=" * 55)

    dates = beta_df.index

    # Dictionaries to accumulate monthly results
    port_returns = {}    # {date: return}
    weight_store = {}    # {date: weight_series}

    skipped = 0
    traded  = 0

    # For loop — one iteration per month
    for i in range(len(dates) - 1):

        signal_date = dates[i]       # beta signal from this month
        return_date = dates[i + 1]   # actual return next month

        beta_row = beta_df.loc[signal_date]

        # Condition: skip if return date not in stock data
        if return_date not in stock_returns.index:
            skipped += 1
            continue

        ret_row = stock_returns.loc[return_date]

        # Get weights for this month
        long_w, short_w = compute_bab_weights(beta_row)

        # Condition: skip if insufficient data
        if long_w.empty or short_w.empty:
            skipped += 1
            continue

        # ── Beta-scale each leg to weighted-average beta = 1 ──
        long_betas  = beta_row.reindex(long_w.index).dropna()
        short_betas = beta_row.reindex(
            short_w.index.intersection(beta_row.index)
        ).dropna()

        long_beta_port  = (long_w.reindex(long_betas.index).fillna(0)
                           * long_betas).sum()
        short_beta_port = (
            (-short_w.reindex(short_betas.index).fillna(0))
            * short_betas
        ).sum()

        # Condition: avoid near-zero division
        if abs(long_beta_port) < 1e-6 or abs(short_beta_port) < 1e-6:
            skipped += 1
            continue

        long_scale  = 1.0 / long_beta_port
        short_scale = 1.0 / short_beta_port

        # ── Compute portfolio return ───────────────────────────
        long_ret  = (long_w  * ret_row.reindex(long_w.index).fillna(0)).sum()
        short_ret = (short_w * ret_row.reindex(short_w.index).fillna(0)).sum()

        # BAB return = (1/β_L) * R_L  -  (1/β_H) * R_H
        bab_return = long_scale * long_ret - short_scale * (-short_ret)

        # Store in dictionaries
        port_returns[return_date] = bab_return

        combined = pd.concat([long_w * long_scale, short_w * short_scale])
        weight_store[return_date] = combined

        traded += 1

    print(f"   ✅ Months traded  : {traded}")
    print(f"   ⚠️  Months skipped : {skipped}  (insufficient data)")

    # Build output Series and DataFrame from dictionaries
    port_series = pd.Series(port_returns, name="BAB_Return")
    port_series.index.name = "Date"

    weight_df = pd.DataFrame(weight_store).T
    weight_df.index.name = "Date"

    return port_series, weight_df


# ============================================================
# STEP 5 — EQUITY CURVE
# ============================================================

def compute_equity_curve(
    port_returns : Series,
    initial      : float = 1.0,
) -> Series:
    """
    Converts monthly returns to cumulative wealth index.

    Example: +5% month gives multiplier 1.05.
    After n months: wealth = initial × Π(1 + r_t)

    Uses: pandas cumprod
    """
    equity      = (1 + port_returns).cumprod() * initial
    equity.name = "BAB_Equity"
    return equity


# ============================================================
# STEP 6 — PERFORMANCE SUMMARY
# ============================================================

def performance_summary(
    port_returns : Series,
    rf           : Series | None = None,
) -> None:
    """
    Prints an annualised performance table.
    Uses: list of tuples, for loop, conditions
    """
    print("\n" + "=" * 55)
    print("STEP 5: Performance Summary")
    print("=" * 55)

    n     = len(port_returns)
    years = n / 12

    arith_mean = port_returns.mean() * 12
    geo_mean   = (1 + port_returns).prod() ** (1 / years) - 1
    std_ann    = port_returns.std() * np.sqrt(12)
    total_ret  = (1 + port_returns).prod() - 1

    # Sharpe ratio
    if rf is not None and len(rf) > 0:
        rf_aligned = rf.reindex(port_returns.index).fillna(0)
        excess     = port_returns - rf_aligned
        sharpe     = (excess.mean() / excess.std()) * np.sqrt(12)
    else:
        sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(12)

    # Max drawdown
    equity      = compute_equity_curve(port_returns)
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # List of (label, value) tuples — for loop to print
    metrics = [
        ("Months of data",              str(n)),
        ("Annualised arithmetic mean",  f"{arith_mean:.2%}"),
        ("Annualised geometric mean",   f"{geo_mean:.2%}"),
        ("Annualised std deviation",    f"{std_ann:.2%}"),
        ("Annualised Sharpe ratio",     f"{sharpe:.2f}"),
        ("Max drawdown",                f"{max_dd:.2%}"),
        ("Total cumulative return",     f"{total_ret:.2%}"),
        ("Final equity (started 1.0)",  f"{equity.iloc[-1]:.2f}"),
    ]

    print(f"\n   {'Metric':<35} {'Value':>12}")
    print(f"   {'-'*47}")
    for label, value in metrics:
        print(f"   {label:<35} {value:>12}")


# ============================================================
# STEP 7 — PREVIEW
# ============================================================

def preview_outputs(
    port_returns : Series,
    weight_df    : DataFrame,
    equity       : Series,
) -> None:
    print("\n" + "=" * 55)
    print("STEP 4: Data Preview")
    print("=" * 55)

    print(f"\nPortfolio returns — first {PREVIEW_ROWS} months:")
    print(port_returns.head(PREVIEW_ROWS).round(4).to_string())

    print(f"\nPortfolio returns — last {PREVIEW_ROWS} months:")
    print(port_returns.tail(PREVIEW_ROWS).round(4).to_string())

    print(f"\nEquity curve — last {PREVIEW_ROWS} months:")
    print(equity.tail(PREVIEW_ROWS).round(4).to_string())

    if not weight_df.empty:
        sample_cols = weight_df.columns[:4].tolist()
        print(f"\nWeights (first 4 tickers, first {PREVIEW_ROWS} months):")
        print(weight_df[sample_cols].head(PREVIEW_ROWS).round(4).to_string())


# ============================================================
# STEP 8 — SAVE
# ============================================================

def save_outputs(
    port_returns : Series,
    weight_df    : DataFrame,
    equity       : Series,
) -> None:
    print("\n" + "=" * 55)
    print("STEP 6: Saving outputs")
    print("=" * 55)

    os.makedirs("data/processed", exist_ok=True)

    try:
        port_returns.to_csv(PORTFOLIO_OUT, header=True)
        print(f"   ✅ Portfolio returns : {PORTFOLIO_OUT}")

        weight_df.to_csv(WEIGHTS_OUT)
        print(f"   ✅ Portfolio weights : {WEIGHTS_OUT}")

        equity.to_csv(EQUITY_OUT, header=True)
        print(f"   ✅ Equity curve      : {EQUITY_OUT}")

    except Exception as e:
        print(f"   ❌ ERROR saving: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("PERSON 3 — BAB PORTFOLIO CONSTRUCTION & BACKTESTING")
    print("=" * 55)

    # Step 1: Load
    stock_returns, market_ret, rf = load_data()

    # Step 2: Betas (Person 2)
    beta_df = compute_betas(stock_returns, market_ret, rf)

    # Step 3: Build portfolio
    port_returns, weight_df = build_portfolio(beta_df, stock_returns)

    if port_returns.empty:
        print("\n❌ No portfolio returns generated — check data inputs.")
        sys.exit(1)

    # Step 4: Equity curve
    equity = compute_equity_curve(port_returns)

    # Step 5: Preview
    preview_outputs(port_returns, weight_df, equity)

    # Step 6: Performance summary
    performance_summary(port_returns, rf if len(rf) > 0 else None)

    # Step 7: Save
    save_outputs(port_returns, weight_df, equity)

    print("\n" + "=" * 55)
    print("FOR YOUR TEAMMATES")
    print("=" * 55)
    print("Person 4 — load regression input:")
    print(f"   r = pd.read_csv('{PORTFOLIO_OUT}',")
    print(f"       index_col='Date', parse_dates=True)['BAB_Return']")
    print("\nPerson 5 — load equity curve:")
    print(f"   e = pd.read_csv('{EQUITY_OUT}',")
    print(f"       index_col='Date', parse_dates=True)['BAB_Equity']")
    print("=" * 55)

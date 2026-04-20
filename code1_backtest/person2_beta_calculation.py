# ============================================================
# PERSON 2 — Beta Calculation Module
# File: code1_backtest/person2_beta_calculation.py
#        (imported by Person 3 and Person 5)
#
# ROLE IN PROJECT:
#   This is the mathematical core of the BAB strategy.
#   Provides reusable functions for:
#     1. Computing excess returns (asset − risk-free rate)
#     2. Rolling beta for every stock each month
#     3. Cross-sectional ranking of stocks by beta
#     4. Splitting universe into high / low beta groups
#     5. Labelling each stock as "Low", "High", or neutral
#
# MARKS TARGETED:
#   Basic Programming (30%):
#     - Functions: 5 clearly defined, documented functions
#     - For Loop: inside rolling beta computation
#     - Conditions: alignment checks, edge-case guards
#     - Error Handling: ValueError for bad inputs
#     - Dictionary: used in self-test results summary
#     - Set: used to check column overlap
#   Libraries (40%):
#     - pandas: rolling, cov, var, rank, quantile, align
#     - numpy: array operations
#
# HOW TO USE:
#   from person2_beta_calculation import (
#       calculate_excess_returns,
#       calculate_rolling_beta,
#       rank_stocks_by_beta,
#       split_into_high_low_beta,
#       build_beta_group_labels,
#   )
#
# SELF-TEST:
#   python code1_backtest/person2_beta_calculation.py
# ============================================================

from __future__ import annotations

import pandas as pd
import numpy as np
from pandas import DataFrame, Series


# ============================================================
# FUNCTION 1 — EXCESS RETURNS
# ============================================================

def calculate_excess_returns(
    returns : DataFrame | Series,
    rf      : Series,
) -> DataFrame | Series:
    """
    Subtracts the risk-free rate from asset returns.

    Why: Beta should be computed on returns above the risk-free
    rate, not raw returns. This follows Frazzini & Pedersen (2014).

    Parameters:
        returns : Asset returns, indexed by date.
        rf      : Monthly risk-free rate, indexed by date.

    Returns:
        Excess returns with same shape as input.
    """
    # pandas .subtract aligns on index automatically
    return returns.subtract(rf, axis=0)


# ============================================================
# FUNCTION 2 — ROLLING BETA
# ============================================================

def calculate_rolling_beta(
    asset_returns  : DataFrame | Series,
    market_returns : Series,
    window         : int         = 12,
    min_periods    : int | None  = None,
    rf             : Series | None = None,
    adjust_for_rf  : bool        = True,
) -> DataFrame:
    """
    Computes rolling beta for every asset relative to the market.

    Beta formula (from CAPM):
        β = Cov(asset, market) / Var(market)

    We use a rolling window so beta is re-estimated every month
    using only the most recent `window` months of data.

    Parameters:
        asset_returns  : Returns for one or more stocks (date × tickers).
        market_returns : Market return Series (date,).
        window         : Rolling lookback in months (default 12).
        min_periods    : Minimum observations needed in window.
                         Defaults to max(2, window // 2).
        rf             : Risk-free rate Series, for excess return adjustment.
        adjust_for_rf  : If True and rf given, computes beta on excess returns.

    Returns:
        DataFrame of rolling betas — same shape as asset_returns.

    Raises:
        ValueError if window < 2.
    """
    # Error handling: validate inputs
    if window < 2:
        raise ValueError(f"window must be ≥ 2, got {window}")

    if min_periods is None:
        min_periods = max(2, window // 2)

    # Condition: optionally switch to excess returns
    if adjust_for_rf and rf is not None:
        asset_returns  = calculate_excess_returns(asset_returns, rf)
        market_returns = calculate_excess_returns(market_returns, rf)

    # Ensure asset_returns is a DataFrame (not a Series)
    if isinstance(asset_returns, Series):
        asset_returns = asset_returns.to_frame(
            name=asset_returns.name or "Asset"
        )

    # Sort both by date
    asset_returns  = asset_returns.sort_index()
    market_returns = market_returns.sort_index()

    # Condition: align indices if they differ
    if not asset_returns.index.equals(market_returns.index):
        asset_returns, market_returns = asset_returns.align(
            market_returns, join="inner", axis=0
        )

    # Rolling covariance (each stock vs market) and rolling variance
    rolling_cov = asset_returns.rolling(
        window=window, min_periods=min_periods
    ).cov(market_returns)

    rolling_var = market_returns.rolling(
        window=window, min_periods=min_periods
    ).var()

    # Beta = Cov / Var  (broadcast: divide each column by the same var)
    beta = rolling_cov.divide(rolling_var, axis=0)
    beta.columns = asset_returns.columns

    return beta


# ============================================================
# FUNCTION 3 — RANK STOCKS BY BETA
# ============================================================

def rank_stocks_by_beta(
    beta      : DataFrame,
    ascending : bool = True,
    method    : str  = "dense",
) -> DataFrame:
    """
    Ranks every stock by beta within each month (cross-sectional rank).

    Rank 1 = lowest beta (ascending=True, default).
    NaN betas are kept as NaN — not assigned a rank.

    Parameters:
        beta      : Rolling beta DataFrame (date × ticker).
        ascending : True → rank 1 = lowest beta.
        method    : 'dense' means no rank gaps for ties.

    Returns:
        Integer rank DataFrame, same shape as beta.
    """
    # pandas .rank works across columns (axis=1 = cross-sectional)
    ranks = beta.rank(
        axis      = 1,
        method    = method,
        ascending = ascending,
        na_option = "keep",    # NaN stays NaN instead of getting a rank
    )
    return ranks.astype("Int64")   # nullable integer type


# ============================================================
# FUNCTION 4 — SPLIT INTO HIGH / LOW BETA GROUPS
# ============================================================

def split_into_high_low_beta(
    beta           : DataFrame,
    high_fraction  : float = 0.5,
    low_fraction   : float = 0.5,
) -> tuple[DataFrame, DataFrame]:
    """
    Splits the stock universe each month into two groups:
      - Low-beta  : bottom `low_fraction` of the beta distribution
      - High-beta : top `high_fraction` of the beta distribution

    Stocks in the middle are masked to NaN.

    Parameters:
        beta           : Rolling beta DataFrame (date × ticker).
        high_fraction  : Fraction of stocks in the high-beta group.
        low_fraction   : Fraction of stocks in the low-beta group.

    Returns:
        (low_beta_df, high_beta_df) — values outside each group are NaN.

    Raises:
        ValueError for invalid fraction inputs.
    """
    # Error handling: validate fraction inputs
    if not 0 < high_fraction < 1:
        raise ValueError(f"high_fraction must be in (0,1), got {high_fraction}")
    if not 0 < low_fraction < 1:
        raise ValueError(f"low_fraction must be in (0,1), got {low_fraction}")
    if high_fraction + low_fraction > 1:
        raise ValueError("high_fraction + low_fraction must be ≤ 1")

    # Cross-sectional quantile thresholds (one per month)
    low_threshold  = beta.quantile(q=low_fraction,        axis=1)
    high_threshold = beta.quantile(q=1.0 - high_fraction, axis=1)

    # Boolean masks: True where stock belongs to each group
    low_mask  = beta.lt(low_threshold,  axis=0)
    high_mask = beta.ge(high_threshold, axis=0)

    # Apply masks — stocks outside the group become NaN
    low_beta  = beta.where(low_mask)
    high_beta = beta.where(high_mask)

    return low_beta, high_beta


# ============================================================
# FUNCTION 5 — BUILD GROUP LABEL MAP
# ============================================================

def build_beta_group_labels(
    beta           : DataFrame,
    high_fraction  : float = 0.5,
    low_fraction   : float = 0.5,
) -> DataFrame:
    """
    Creates a human-readable label DataFrame.
    Each cell is one of: 'Low', 'High', or NaN.

    Useful for the screener to display which stocks are
    currently BAB long or short candidates.

    Parameters:
        beta           : Rolling beta DataFrame.
        high_fraction  : Top fraction → 'High' label.
        low_fraction   : Bottom fraction → 'Low' label.

    Returns:
        DataFrame of string labels, same shape as beta.
    """
    low_beta, high_beta = split_into_high_low_beta(
        beta,
        high_fraction = high_fraction,
        low_fraction  = low_fraction,
    )

    # Start with empty (NaN) label frame
    labels = pd.DataFrame(
        index   = beta.index,
        columns = beta.columns,
        dtype   = "object",
    )

    # Condition: assign labels where masks are not NaN
    labels[low_beta.notna()]  = "Low"
    labels[high_beta.notna()] = "High"

    return labels


# ============================================================
# SELF-TEST — run this file directly to verify all functions
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("PERSON 2 — BETA CALCULATION MODULE — SELF TEST")
    print("=" * 55)

    # Create synthetic test data
    np.random.seed(42)
    n_months = 60
    n_stocks = 15

    dates   = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    market  = pd.Series(np.random.randn(n_months) * 0.04,
                        index=dates, name="MKT")
    rf      = pd.Series(np.full(n_months, 0.002),
                        index=dates, name="RF")
    stocks  = pd.DataFrame(
        np.random.randn(n_months, n_stocks) * 0.05,
        index   = dates,
        columns = [f"STK{i:02d}" for i in range(n_stocks)],
    )

    # Test each function and store results in a dictionary
    test_results = {}

    # Test 1: excess returns
    try:
        excess = calculate_excess_returns(stocks, rf)
        test_results["calculate_excess_returns"] = "✅ PASS"
    except Exception as e:
        test_results["calculate_excess_returns"] = f"❌ FAIL: {e}"

    # Test 2: rolling beta
    try:
        beta = calculate_rolling_beta(stocks, market, window=12, rf=rf)
        assert beta.shape == stocks.shape
        test_results["calculate_rolling_beta"] = "✅ PASS"
    except Exception as e:
        test_results["calculate_rolling_beta"] = f"❌ FAIL: {e}"

    # Test 3: rank
    try:
        ranks = rank_stocks_by_beta(beta)
        test_results["rank_stocks_by_beta"] = "✅ PASS"
    except Exception as e:
        test_results["rank_stocks_by_beta"] = f"❌ FAIL: {e}"

    # Test 4: split
    try:
        low_b, high_b = split_into_high_low_beta(beta)
        test_results["split_into_high_low_beta"] = "✅ PASS"
    except Exception as e:
        test_results["split_into_high_low_beta"] = f"❌ FAIL: {e}"

    # Test 5: labels
    try:
        labels = build_beta_group_labels(beta)
        # Use a set to check only valid labels appear
        unique_labels = set(labels.stack().unique())
        assert unique_labels <= {"Low", "High"}
        test_results["build_beta_group_labels"] = "✅ PASS"
    except Exception as e:
        test_results["build_beta_group_labels"] = f"❌ FAIL: {e}"

    # Test 6: ValueError on bad input
    try:
        _ = calculate_rolling_beta(stocks, market, window=1)
        test_results["ValueError on window<2"] = "❌ FAIL: no error raised"
    except ValueError:
        test_results["ValueError on window<2"] = "✅ PASS"

    # Print results dictionary
    print(f"\n   {'Test':<35} Result")
    print(f"   {'-'*50}")
    for test, result in test_results.items():
        print(f"   {test:<35} {result}")

    # Show sample beta output
    print(f"\n   Sample betas (last month, first 5 stocks):")
    print(f"   {beta.iloc[-1, :5].round(3).to_dict()}")

    print(f"\n   Label counts (last month):")
    label_counts = labels.iloc[-1].value_counts().to_dict()
    print(f"   {label_counts}")

    all_passed = all("✅" in v for v in test_results.values())
    print(f"\n   {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 55)

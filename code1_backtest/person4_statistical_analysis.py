# ============================================================
# PERSON 4 — Statistical Analysis Module
# File: code1_backtest/person4_statistical_analysis.py
#
# ROLE IN PROJECT:
#   Runs all the statistical tests and regressions that answer
#   the core research question:
#   "Does the BAB strategy generate significant positive alpha?"
#
#   Tests performed:
#     1. Descriptive statistics (mean, std, Sharpe, skew, kurtosis)
#     2. Normality test         (Shapiro-Wilk)
#     3. Stationarity test      (Augmented Dickey-Fuller)
#     4. Mean/median tests      (t-test + Wilcoxon signed-rank)
#     5. Sharpe ratio comparison
#     6. Simple OLS regression  (BAB ~ MKT)   ← answers research question
#     7. FF3 multiple regression (BAB ~ MKT + SMB + HML) ← robustness
#     8. Diagnostic tests       (Breusch-Pagan, Durbin-Watson, VIF)
#   BONUS: Sends a Telegram alert with key results
#
# MARKS TARGETED:
#   Basic Programming (30%):
#     - Functions: every analysis step in its own function
#     - Dictionary: results stored as dicts before saving
#     - For Loop: iterates over test series, factor columns
#     - Conditions: significance level decisions, star notation
#     - Error Handling: try/except on every model fit and file op
#     - List: metrics list for tabular printing
#   Libraries (40%) — HIGH VALUE:
#     - statsmodels: OLS, summary, het_breuschpagan, durbin_watson, VIF
#     - scipy.stats: shapiro, adfuller, ttest_1samp, wilcoxon, pearsonr
#     - pandas: DataFrame, to_csv, describe
#     - telegram (python-telegram-bot): sends alert to Telegram
#
# INPUTS:
#   data/processed/aqr_returns.csv            (Person 1)
#   data/processed/bab_portfolio_returns.csv  (Person 3) [optional]
#
# OUTPUTS:
#   data/results/descriptive_stats.csv
#   data/results/hypothesis_tests.csv
#   data/results/regression_simple.csv
#   data/results/regression_ff3.csv
#
# HOW TO RUN:
#   python code1_backtest/person4_statistical_analysis.py
#
# TELEGRAM SETUP (optional):
#   1. Message @BotFather on Telegram → create bot → copy token
#   2. Get your chat ID from @userinfobot
#   3. Fill in TELEGRAM_TOKEN and TELEGRAM_CHAT_ID below
# ============================================================

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

AQR_PATH      = "data/processed/aqr_returns.csv"
BAB_PORT_PATH = "data/processed/bab_portfolio_returns.csv"
RESULTS_DIR   = "data/results"

# Set to True to use Person 3's backtest, False to use AQR's BAB column
USE_AQR_BAB   = True

ALPHA_LEVEL   = 0.05   # significance level for all tests

# ── Telegram settings ─────────────────────────────────────────
# Fill these in to receive a Telegram message with key results
TELEGRAM_TOKEN   = "YOUR_BOT_TOKEN_HERE"    # from @BotFather
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"      # from @userinfobot
SEND_TELEGRAM    = False   # Set True to enable Telegram alerts


# ============================================================
# HELPER — SIGNIFICANCE STARS
# ============================================================

def stars(p: float) -> str:
    """Returns star notation based on p-value. Uses conditions."""
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Loads AQR factor data. Optionally replaces BAB column
    with Person 3's backtested returns.
    Uses: try/except, conditions
    """
    print("\n" + "=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    if not os.path.exists(AQR_PATH):
        raise FileNotFoundError(
            f"AQR data not found: {AQR_PATH}\n"
            "Run person1_data_loader.py first."
        )

    try:
        df = pd.read_csv(AQR_PATH, index_col="Date", parse_dates=True)
        print(f"   ✅ AQR loaded  : {df.shape[0]} months × "
              f"{df.shape[1]} columns")
        print(f"      Columns    : {list(df.columns)}")
        print(f"      Date range : {df.index[0].date()} → "
              f"{df.index[-1].date()}")
    except Exception as e:
        raise RuntimeError(f"Failed to load AQR data: {e}")

    # Condition: optionally swap in Person 3's backtest returns
    if not USE_AQR_BAB and os.path.exists(BAB_PORT_PATH):
        try:
            bab_port = pd.read_csv(
                BAB_PORT_PATH, index_col="Date", parse_dates=True
            )["BAB_Return"]
            df["BAB"] = bab_port.reindex(df.index)
            print(f"   ✅ Replaced BAB with Person 3 backtest")
        except Exception as e:
            print(f"   ⚠️  Could not load backtest: {e}")

    # Drop rows with missing BAB or MKT (needed for regression)
    df = df.dropna(subset=["BAB", "MKT"])
    print(f"   ✅ After cleaning: {len(df)} months")

    return df


# ============================================================
# STEP 2 — DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the full descriptive stats table.
    Arithmetic mean, geometric mean, std → annualised (×12 or ×√12).
    Median, IQR → kept monthly (not annualised).
    Uses: for loop, dictionary, list
    """
    print("\n" + "=" * 60)
    print("STEP 2: Descriptive Statistics")
    print("=" * 60)

    # List of columns to analyse
    cols = [c for c in ["BAB", "MKT", "SMB", "HML"] if c in df.columns]

    # Dictionary to collect results
    records = {}

    # For loop over each column
    for col in cols:
        s    = df[col].dropna()
        yrs  = len(s) / 12

        # Build a dictionary for this column's stats
        records[col] = {
            "N (months)"               : len(s),
            "Arith. Mean (ann.)"       : round(s.mean() * 12, 4),
            "Geom. Mean (ann.)"        : round((1 + s).prod() ** (1/yrs) - 1, 4),
            "Std Dev (ann.)"           : round(s.std() * np.sqrt(12), 4),
            "Median (monthly)"         : round(s.median(), 4),
            "IQR (monthly)"            : round(s.quantile(0.75) - s.quantile(0.25), 4),
            "Skewness"                 : round(s.skew(), 4),
            "Excess Kurtosis"          : round(s.kurt(), 4),
            "Min"                      : round(s.min(), 4),
            "Max"                      : round(s.max(), 4),
        }

    desc_df = pd.DataFrame(records)
    print(f"\n{desc_df.to_string()}")
    return desc_df


# ============================================================
# STEP 3 — HYPOTHESIS TESTS
# ============================================================

def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs four tests for BAB, MKT, and the BAB−MKT difference:
      1. Shapiro-Wilk  → is the series normally distributed?
      2. ADF           → is the series stationary?
      3. t-test        → is the mean significantly ≠ 0?
      4. Wilcoxon      → is the median significantly ≠ 0?
    Uses: for loop, dictionary, conditions, try/except
    """
    print("\n" + "=" * 60)
    print("STEP 3: Hypothesis Tests")
    print("=" * 60)

    bab  = df["BAB"].dropna()
    mkt  = df["MKT"].reindex(bab.index).dropna()
    diff = (bab - mkt).dropna()

    # List of (label, series) pairs to test
    test_series = [
        ("BAB",       bab),
        ("MKT",       mkt),
        ("BAB − MKT", diff),
    ]

    results = {}   # Dictionary of test results

    # For loop over each series
    for label, s in test_series:
        try:
            sw_stat, sw_p   = stats.shapiro(s)
            adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
            t_stat,  t_p    = stats.ttest_1samp(s, 0)
            w_stat,  w_p    = stats.wilcoxon(s)

            # Dictionary for this series
            results[label] = {
                "Shapiro-Wilk stat" : round(sw_stat,  4),
                "Shapiro-Wilk p"    : f"{sw_p:.3e}",
                "ADF stat"          : round(adf_stat,  4),
                "ADF p"             : f"{adf_p:.3e}",
                "t-test stat"       : round(t_stat,   4),
                "t-test p"          : f"{t_p:.3e}",
                "Wilcoxon stat"     : round(w_stat,   4),
                "Wilcoxon p"        : f"{w_p:.3e}",
            }
        except Exception as e:
            print(f"   ⚠️  Error testing {label}: {e}")

    test_df = pd.DataFrame(results)
    print(f"\n{test_df.to_string()}")

    # Plain-language interpretation
    print(f"\n   INTERPRETATION:")
    for label, s in test_series[:2]:
        _, sw_p = stats.shapiro(s)
        _, t_p  = stats.ttest_1samp(s, 0)
        # Condition: interpret significance
        norm    = "NOT normal" if sw_p < ALPHA_LEVEL else "normal"
        signif  = "significant ≠ 0" if t_p < ALPHA_LEVEL else "not significant"
        print(f"   {label}: distribution is {norm} | mean is {signif}")

    _, t_diff = stats.ttest_1samp(diff, 0)
    _, w_diff = stats.wilcoxon(diff)
    verdict   = "significant" if (t_diff < ALPHA_LEVEL or w_diff < ALPHA_LEVEL) \
                else "NOT significant"
    print(f"   BAB − MKT outperformance: {verdict}")

    return test_df


# ============================================================
# STEP 4 — SHARPE RATIO
# ============================================================

def compute_sharpe(df: pd.DataFrame) -> dict:
    """
    Annualised Sharpe ratio for BAB and MKT.
    Uses: dictionary, for loop, conditions
    """
    print("\n" + "=" * 60)
    print("STEP 4: Sharpe Ratios")
    print("=" * 60)

    rf = df["RF"].reindex(df.index).fillna(0) \
         if "RF" in df.columns else pd.Series(0.0, index=df.index)

    sharpe_dict = {}   # Dictionary: column → Sharpe ratio

    # For loop over columns of interest
    for col in ["BAB", "MKT"]:
        if col not in df.columns:
            continue
        excess = df[col] - rf
        sharpe = (excess.mean() / excess.std()) * np.sqrt(12)
        sharpe_dict[col] = round(sharpe, 4)
        print(f"   {col} Sharpe ratio (ann.) : {sharpe:.4f}")

    return sharpe_dict


# ============================================================
# STEP 5 — SIMPLE OLS REGRESSION  (BAB ~ MKT)
# ============================================================

def simple_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates: BAB_t = α + β*MKT_t + ε_t

    A statistically significant positive α means the BAB
    strategy earns excess return unexplained by market risk.
    Uses: try/except, dictionary, conditions
    """
    print("\n" + "=" * 60)
    print("STEP 5: Simple OLS Regression  BAB ~ MKT")
    print("=" * 60)

    try:
        bab = df["BAB"].dropna()
        mkt = df["MKT"].reindex(bab.index).dropna()
        idx = bab.index.intersection(mkt.index)

        y = bab.loc[idx]
        X = sm.add_constant(mkt.loc[idx])   # adds intercept column

        model = sm.OLS(y, X).fit()
        print(f"\n{model.summary()}")

        # Diagnostic tests
        dw           = durbin_watson(model.resid)
        _, bp_p, _, _= het_breuschpagan(model.resid, model.model.exog)
        pearson_r, _ = stats.pearsonr(X["MKT"], y)

        print(f"\n   DIAGNOSTICS:")
        print(f"   Durbin-Watson       : {dw:.4f}  "
              f"(~2 = no serial autocorrelation)")
        print(f"   Breusch-Pagan p     : {bp_p:.4f}  "
              f"({'homoscedastic ✅' if bp_p > 0.05 else 'heteroscedastic ⚠️'})")
        print(f"   Pearson r (MKT,BAB) : {pearson_r:.4f}")

        # Dictionary of key results
        results_dict = {
            "alpha"           : round(model.params["const"],      6),
            "alpha_pvalue"    : round(model.pvalues["const"],     6),
            "alpha_stars"     : stars(model.pvalues["const"]),
            "beta_MKT"        : round(model.params["MKT"],        6),
            "beta_MKT_pvalue" : round(model.pvalues["MKT"],       6),
            "R2"              : round(model.rsquared,              4),
            "R2_adj"          : round(model.rsquared_adj,          4),
            "N"               : int(model.nobs),
            "DW"              : round(dw,                          4),
            "BP_p"            : round(bp_p,                       4),
        }

        return pd.DataFrame([results_dict]), model

    except Exception as e:
        print(f"   ❌ Regression failed: {e}")
        return pd.DataFrame(), None


# ============================================================
# STEP 6 — FF3 MULTIPLE REGRESSION  (BAB ~ MKT + SMB + HML)
# ============================================================

def multiple_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates: BAB_t = α + β1*MKT_t + β2*SMB_t + β3*HML_t + ε_t

    Robustness check: if alpha survives adding size (SMB) and
    value (HML) factors, it is truly strategy-specific.
    Uses: try/except, dictionary, for loop (VIF), conditions
    """
    print("\n" + "=" * 60)
    print("STEP 6: Multiple OLS Regression  BAB ~ MKT + SMB + HML  [FF3]")
    print("=" * 60)

    required_cols = ["BAB", "MKT", "SMB", "HML"]
    missing_cols  = [c for c in required_cols if c not in df.columns]

    # Condition: check all columns present
    if missing_cols:
        print(f"   ⚠️  Missing columns: {missing_cols} — skipping FF3")
        return pd.DataFrame(), None

    try:
        sub = df[required_cols].dropna()
        y   = sub["BAB"]
        X   = sm.add_constant(sub[["MKT", "SMB", "HML"]])

        model = sm.OLS(y, X).fit()
        print(f"\n{model.summary()}")

        # Diagnostics
        dw           = durbin_watson(model.resid)
        _, bp_p, _, _= het_breuschpagan(model.resid, model.model.exog)

        # VIF — for loop over each factor column
        vif_dict = {}
        for i in range(1, X.shape[1]):   # skip constant at index 0
            col_name       = X.columns[i]
            vif_val        = variance_inflation_factor(X.values, i)
            vif_dict[col_name] = round(vif_val, 4)

        print(f"\n   DIAGNOSTICS:")
        print(f"   Durbin-Watson   : {dw:.4f}")
        print(f"   Breusch-Pagan p : {bp_p:.4f}  "
              f"({'homoscedastic ✅' if bp_p > 0.05 else 'heteroscedastic ⚠️'})")
        print(f"   VIF values:")
        for factor, vif in vif_dict.items():
            flag = "✅ OK" if vif < 5 else "⚠️ high"
            print(f"      {factor}: {vif:.2f}  {flag}")

        # Dictionary of results
        results_dict = {
            "alpha"           : round(model.params["const"],  6),
            "alpha_pvalue"    : round(model.pvalues["const"], 6),
            "alpha_stars"     : stars(model.pvalues["const"]),
            "beta_MKT"        : round(model.params["MKT"],    6),
            "beta_MKT_pvalue" : round(model.pvalues["MKT"],   6),
            "beta_SMB"        : round(model.params["SMB"],    6),
            "beta_SMB_pvalue" : round(model.pvalues["SMB"],   6),
            "beta_SMB_stars"  : stars(model.pvalues["SMB"]),
            "beta_HML"        : round(model.params["HML"],    6),
            "beta_HML_pvalue" : round(model.pvalues["HML"],   6),
            "beta_HML_stars"  : stars(model.pvalues["HML"]),
            "R2"              : round(model.rsquared,          4),
            "R2_adj"          : round(model.rsquared_adj,      4),
            "N"               : int(model.nobs),
            "DW"              : round(dw,                      4),
            "BP_p"            : round(bp_p,                   4),
        }

        return pd.DataFrame([results_dict]), model

    except Exception as e:
        print(f"   ❌ FF3 Regression failed: {e}")
        return pd.DataFrame(), None


# ============================================================
# BONUS — TELEGRAM ALERT
# ============================================================

def send_telegram_alert(simple_results: pd.DataFrame, sharpe_dict: dict) -> None:
    """
    Sends a Telegram message with the key regression results.
    Uses: try/except, conditions, f-strings
    Libraries: requests (HTTP POST to Telegram API)
    """
    if not SEND_TELEGRAM:
        print("\n   ℹ️  Telegram alerts disabled. Set SEND_TELEGRAM=True to enable.")
        return

    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("\n   ⚠️  Please set your TELEGRAM_TOKEN and TELEGRAM_CHAT_ID")
        return

    print("\n" + "=" * 60)
    print("BONUS: Sending Telegram Alert")
    print("=" * 60)

    try:
        import requests as req   # using requests library for HTTP

        # Condition: check results available
        if simple_results.empty:
            print("   ⚠️  No results to send")
            return

        row = simple_results.iloc[0]

        # Build message string
        msg = (
            f"📊 *BAB Strategy Analysis Complete*\n\n"
            f"*Simple Regression (BAB ~ MKT):*\n"
            f"  Alpha  : {float(row['alpha']):.4f} {row['alpha_stars']}\n"
            f"  p-value: {float(row['alpha_pvalue']):.4e}\n"
            f"  β(MKT) : {float(row['beta_MKT']):.4f}\n"
            f"  R²     : {float(row['R2']):.4f}\n\n"
            f"*Sharpe Ratios (annualised):*\n"
            f"  BAB    : {sharpe_dict.get('BAB', 'N/A')}\n"
            f"  Market : {sharpe_dict.get('MKT', 'N/A')}\n\n"
            f"*Verdict:*\n"
            f"  Alpha {'significant ✅' if float(row['alpha_pvalue']) < 0.05 else 'not significant ❌'}"
        )

        url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id"    : TELEGRAM_CHAT_ID,
            "text"       : msg,
            "parse_mode" : "Markdown",
        }

        response = req.post(url, data=payload, timeout=10)

        # Condition: check HTTP response
        if response.status_code == 200:
            print(f"   ✅ Telegram alert sent successfully")
        else:
            print(f"   ⚠️  Telegram returned status {response.status_code}")

    except Exception as e:
        print(f"   ❌ Telegram error: {e}")


# ============================================================
# STEP 7 — SAVE RESULTS
# ============================================================

def save_results(
    desc_df   : pd.DataFrame,
    test_df   : pd.DataFrame,
    simple_df : pd.DataFrame,
    ff3_df    : pd.DataFrame,
) -> None:
    """
    Saves all result tables as CSV files.
    Uses: for loop over list of (df, name) pairs, try/except
    """
    print("\n" + "=" * 60)
    print("STEP 7: Saving results")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # List of (DataFrame, filename) pairs
    output_pairs = [
        (desc_df,   "descriptive_stats"),
        (test_df,   "hypothesis_tests"),
        (simple_df, "regression_simple"),
        (ff3_df,    "regression_ff3"),
    ]

    # For loop over each pair
    for df, name in output_pairs:
        if df is not None and not df.empty:
            try:
                path = os.path.join(RESULTS_DIR, f"{name}.csv")
                df.to_csv(path)
                print(f"   ✅ Saved: {path}")
            except Exception as e:
                print(f"   ❌ Could not save {name}: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("PERSON 4 — STATISTICAL ANALYSIS MODULE")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. Descriptive statistics
    desc_df = descriptive_statistics(df)

    # 3. Hypothesis tests
    test_df = hypothesis_tests(df)

    # 4. Sharpe ratios
    sharpe_dict = compute_sharpe(df)

    # 5. Simple regression
    simple_df, simple_model = simple_regression(df)

    # 6. FF3 multiple regression
    ff3_df, ff3_model = multiple_regression(df)

    # 7. Telegram alert (bonus library)
    send_telegram_alert(simple_df, sharpe_dict)

    # 8. Save results
    save_results(desc_df, test_df, simple_df, ff3_df)

    print("\n" + "=" * 60)
    print("DONE — Results saved to data/results/")
    print("=" * 60)

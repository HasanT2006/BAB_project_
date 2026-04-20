# ============================================================
# PERSON 5 — Live BAB Screener + All Visualizations
# File: code2_screener/person5_screener_and_charts.py
#
# ROLE IN PROJECT:
#   Part A — Live BAB Stock Screener:
#     Fetches current S&P 500 data, computes live rolling betas,
#     ranks every stock, and outputs a BAB screener table showing
#     which stocks are currently long or short candidates.
#     Exports screener to Google Sheets.
#
#   Part B — All Charts (both Code 1 and Code 2):
#     Chart 1 — Box plots: BAB vs Market monthly returns
#     Chart 2 — QQ plots: normality check
#     Chart 3 — Equity curve: cumulative wealth comparison
#     Chart 4 — Rolling returns bar chart
#     Chart 5 — Beta distribution histogram
#     Chart 6 — Top/bottom beta stocks bar chart
#     Chart 7 — Correlation heatmap (BAB, MKT, SMB, HML)
#
# MARKS TARGETED (HIGH VALUE — 40% of grade):
#   Libraries:
#     - matplotlib + seaborn: 7 publication-quality charts
#     - gspread + oauth2client: Google Sheets export
#     - Person 2's beta module: reused for screener
#     - pandas: all data manipulation
#     - scipy.stats: probplot for QQ chart
#   Basic Programming (30%):
#     - For loop: chart generation loop, screener loop
#     - Dictionary: chart config, screener results
#     - Function: every chart is its own function
#     - Error Handling: try/except on all external calls
#     - Conditions: skip chart if data missing
#
# INPUTS:
#   data/processed/current_returns.csv       (Person 1)
#   data/processed/aqr_returns.csv           (Person 1)
#   data/processed/bab_equity_curve.csv      (Person 3)
#   data/processed/bab_portfolio_returns.csv (Person 3)
#
# OUTPUTS:
#   data/screener/bab_screener_output.csv
#   data/figures/chart_1_boxplots.png
#   data/figures/chart_2_qqplots.png
#   data/figures/chart_3_equity_curve.png
#   data/figures/chart_4_rolling_returns.png
#   data/figures/chart_5_beta_histogram.png
#   data/figures/chart_6_top_bottom_beta.png
#   data/figures/chart_7_correlation_heatmap.png
#   Google Sheets (if configured)
#
# HOW TO RUN:
#   python code2_screener/person5_screener_and_charts.py
#
# GOOGLE SHEETS SETUP (optional):
#   1. Go to Google Cloud Console → create a project
#   2. Enable Google Sheets API and Google Drive API
#   3. Create Service Account → download JSON key file
#   4. Share your Google Sheet with the service account email
#   5. Set GSHEETS_KEY_FILE and GSHEETS_SPREADSHEET_NAME below
# ============================================================

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")    # non-interactive backend — works on any machine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("   ℹ️  seaborn not installed — some charts will use matplotlib only")

warnings.filterwarnings("ignore")

# ── Import Person 2's beta functions ─────────────────────────
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "code1_backtest"
))
try:
    from person2_beta_calculation import (
        calculate_rolling_beta,
        build_beta_group_labels,
    )
    HAS_BETA_MODULE = True
except ImportError:
    HAS_BETA_MODULE = False
    print("   ⚠️  person2_beta_calculation not found — screener disabled")

# ============================================================
# SETTINGS
# ============================================================

RETURNS_PATH  = "data/processed/current_returns.csv"
PRICES_PATH   = "data/processed/current_prices.csv"
AQR_PATH      = "data/processed/aqr_returns.csv"
EQUITY_PATH   = "data/processed/bab_equity_curve.csv"
PORT_PATH     = "data/processed/bab_portfolio_returns.csv"

SCREENER_OUT  = "data/screener/bab_screener_output.csv"
FIGURES_DIR   = "data/figures"

BETA_WINDOW   = 12
MIN_PERIODS   = 6
TOP_N         = 20    # stocks to show in top/bottom chart

# ── Google Sheets settings ────────────────────────────────────
GSHEETS_KEY_FILE        = "google_credentials.json"  # service account JSON
GSHEETS_SPREADSHEET_NAME= "BAB Screener Output"
EXPORT_TO_GSHEETS       = False   # Set True when credentials are ready

# ── Chart style ───────────────────────────────────────────────
COLORS = {
    "bab"    : "#1f77b4",   # blue
    "mkt"    : "#ff7f0e",   # orange
    "smb"    : "#2ca02c",   # green
    "hml"    : "#9467bd",   # purple
    "long"   : "#2ca02c",   # green
    "short"  : "#d62728",   # red
    "neutral": "#7f7f7f",   # grey
}
DPI = 150


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_dirs():
    """Creates output directories if they don't exist."""
    for d in ["data/screener", FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


def save_fig(fig, filename: str) -> None:
    """Saves a matplotlib figure and closes it."""
    path = os.path.join(FIGURES_DIR, filename)
    try:
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"   ✅ Saved: {path}")
    except Exception as e:
        print(f"   ❌ Could not save {filename}: {e}")
    finally:
        plt.close(fig)


def load_aqr() -> pd.DataFrame | None:
    """Loads AQR returns. Returns None if file missing."""
    if not os.path.exists(AQR_PATH):
        print(f"   ⚠️  AQR data not found at {AQR_PATH}")
        return None
    try:
        return pd.read_csv(AQR_PATH, index_col="Date", parse_dates=True)
    except Exception as e:
        print(f"   ⚠️  Could not load AQR: {e}")
        return None


# ============================================================
# PART A — LIVE BAB SCREENER
# ============================================================

def run_screener() -> pd.DataFrame:
    """
    Loads current S&P 500 returns, computes rolling beta for
    every stock, ranks them, and outputs the screener table.
    Uses: for loop, dictionary, conditions, try/except
    """
    print("\n" + "=" * 60)
    print("PART A: Live BAB Stock Screener")
    print("=" * 60)

    if not HAS_BETA_MODULE:
        print("   ❌ Beta module not available")
        return pd.DataFrame()

    if not os.path.exists(RETURNS_PATH):
        print(f"   ❌ {RETURNS_PATH} not found. Run Person 1 first.")
        return pd.DataFrame()

    try:
        returns = pd.read_csv(RETURNS_PATH, index_col="Date", parse_dates=True)
        print(f"   ✅ Loaded : {returns.shape[0]} months × "
              f"{returns.shape[1]} stocks")
    except Exception as e:
        print(f"   ❌ Could not load returns: {e}")
        return pd.DataFrame()

    # Equal-weighted market proxy
    market = returns.mean(axis=1)
    market.name = "EW_Market"

    try:
        beta_df = calculate_rolling_beta(
            asset_returns  = returns,
            market_returns = market,
            window         = BETA_WINDOW,
            min_periods    = MIN_PERIODS,
            adjust_for_rf  = False,
        )
    except Exception as e:
        print(f"   ❌ Beta calculation failed: {e}")
        return pd.DataFrame()

    # Use the latest available row of betas
    latest_beta = beta_df.iloc[-1].dropna().sort_values()
    n = len(latest_beta)

    if n == 0:
        print("   ❌ No valid betas computed")
        return pd.DataFrame()

    # Build screener results in a dictionary then convert
    screener_records = {}

    labels = build_beta_group_labels(
        beta_df.iloc[[-1]], high_fraction=0.5, low_fraction=0.5
    ).iloc[-1].reindex(latest_beta.index)

    # For loop over each stock
    for ticker in latest_beta.index:
        beta_val = latest_beta[ticker]
        pct      = (latest_beta <= beta_val).mean() * 100
        signal   = labels.get(ticker, None)

        screener_records[ticker] = {
            "Beta_12M"   : round(beta_val, 4),
            "Percentile" : round(pct, 1),
            "Signal"     : signal if pd.notna(signal) else "Neutral",
        }

    # Convert dictionary to DataFrame
    screener_df = pd.DataFrame.from_dict(screener_records, orient="index")
    screener_df.index.name = "Ticker"
    screener_df = screener_df.sort_values("Beta_12M")

    # Rank column (1 = lowest beta)
    screener_df["Rank"] = range(1, len(screener_df) + 1)

    # Save to CSV
    try:
        screener_df.to_csv(SCREENER_OUT)
        print(f"   ✅ Screener saved: {SCREENER_OUT}  ({len(screener_df)} stocks)")
    except Exception as e:
        print(f"   ❌ Could not save screener: {e}")

    # Print top and bottom
    print(f"\n   TOP 10 LOWEST BETA — BAB Long Candidates:")
    print(screener_df.head(10)[["Beta_12M", "Percentile", "Signal"]].to_string())
    print(f"\n   TOP 10 HIGHEST BETA — BAB Short Candidates:")
    print(screener_df.tail(10).iloc[::-1][["Beta_12M", "Percentile", "Signal"]].to_string())

    return screener_df


# ============================================================
# GOOGLE SHEETS EXPORT
# ============================================================

def export_to_google_sheets(screener_df: pd.DataFrame) -> None:
    """
    Uploads the screener DataFrame to Google Sheets.
    Uses: try/except, conditions, gspread library
    Libraries: gspread, oauth2client
    """
    if not EXPORT_TO_GSHEETS:
        print("\n   ℹ️  Google Sheets export disabled.")
        print("       Set EXPORT_TO_GSHEETS=True and provide credentials.")
        return

    print("\n" + "=" * 60)
    print("BONUS: Exporting to Google Sheets")
    print("=" * 60)

    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials

        # Condition: check credentials file exists
        if not os.path.exists(GSHEETS_KEY_FILE):
            print(f"   ❌ Credentials file not found: {GSHEETS_KEY_FILE}")
            return

        # Define required API scopes
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]

        # Authenticate with service account
        creds  = ServiceAccountCredentials.from_json_keyfile_name(
            GSHEETS_KEY_FILE, scope
        )
        client = gspread.authorize(creds)

        # Open or create spreadsheet
        try:
            sheet = client.open(GSHEETS_SPREADSHEET_NAME).sheet1
        except gspread.SpreadsheetNotFound:
            sheet = client.create(GSHEETS_SPREADSHEET_NAME).sheet1
            print(f"   ✅ Created new spreadsheet: {GSHEETS_SPREADSHEET_NAME}")

        # Clear existing content
        sheet.clear()

        # Write header row
        header = ["Ticker"] + screener_df.columns.tolist()
        sheet.append_row(header)

        # For loop: write each row of screener data
        for ticker, row in screener_df.iterrows():
            values = [ticker] + [str(v) for v in row.values]
            sheet.append_row(values)

        print(f"   ✅ Exported {len(screener_df)} stocks to Google Sheets")
        print(f"      Spreadsheet: {GSHEETS_SPREADSHEET_NAME}")

    except ImportError:
        print("   ⚠️  gspread not installed. Run: pip install gspread oauth2client")
    except Exception as e:
        print(f"   ❌ Google Sheets error: {e}")


# ============================================================
# PART B — ALL CHARTS
# ============================================================

# ── Chart 1 — Box Plots ──────────────────────────────────────

def chart_boxplots() -> None:
    """BAB vs Market monthly return distributions."""
    print("\n   [Chart 1] Box plots")

    aqr = load_aqr()
    if aqr is None or "BAB" not in aqr.columns:
        return

    bab = aqr["BAB"].dropna()
    mkt = aqr["MKT"].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        [bab.values, mkt.values],
        labels       = ["BAB Strategy", "Market Portfolio"],
        patch_artist = True,
        medianprops  = dict(color="black", linewidth=2),
        whiskerprops = dict(linewidth=1.2),
        flierprops   = dict(marker=".", markersize=4, alpha=0.5),
    )
    bp["boxes"][0].set_facecolor(COLORS["bab"] + "99")
    bp["boxes"][1].set_facecolor(COLORS["mkt"] + "99")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Monthly Excess Return", fontsize=12)
    ax.set_title("Distribution of Monthly Returns: BAB vs Market",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    # Annotate key stats
    for i, (s, label) in enumerate([(bab, "BAB"), (mkt, "Mkt")], 1):
        med = s.median()
        ax.text(i, s.quantile(0.75) + 0.005,
                f"Med: {med:.2%}\nStd: {s.std():.2%}",
                ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

    fig.tight_layout()
    save_fig(fig, "chart_1_boxplots.png")


# ── Chart 2 — QQ Plots ───────────────────────────────────────

def chart_qqplots() -> None:
    """Normality check for BAB and Market returns."""
    print("\n   [Chart 2] QQ plots")

    aqr = load_aqr()
    if aqr is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # For loop over the two series we want to plot
    plot_items = [
        (axes[0], "BAB", "BAB Strategy",    COLORS["bab"]),
        (axes[1], "MKT", "Market Portfolio", COLORS["mkt"]),
    ]

    for ax, col, label, color in plot_items:
        if col not in aqr.columns:
            continue
        s = aqr[col].dropna()

        (osm, osr), (slope, intercept, _) = stats.probplot(s, dist="norm")
        ax.scatter(osm, osr, s=10, alpha=0.6, color=color,
                   label="Observations", zorder=2)

        line_x = np.array([osm.min(), osm.max()])
        ax.plot(line_x, slope * line_x + intercept,
                color="red", linewidth=1.8, label="Normal line", zorder=3)

        ax.set_xlabel("Theoretical Normal Quantiles", fontsize=11)
        ax.set_ylabel("Empirical Quantiles", fontsize=11)
        ax.set_title(f"QQ Plot — {label}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Annotate skew & kurtosis
        ax.text(0.05, 0.95,
                f"Skew: {s.skew():.2f}\nKurt: {s.kurt():.2f}",
                transform=ax.transAxes, va="top",
                fontsize=9, bbox=dict(boxstyle="round", alpha=0.2))

    fig.suptitle("QQ Plots: Are Returns Normally Distributed?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "chart_2_qqplots.png")


# ── Chart 3 — Equity Curve ───────────────────────────────────

def chart_equity_curve() -> None:
    """Cumulative wealth of BAB vs Market (log scale)."""
    print("\n   [Chart 3] Equity curve")

    if not os.path.exists(EQUITY_PATH):
        print("   ⚠️  Equity curve file not found — skipping")
        return

    try:
        bab_eq = pd.read_csv(
            EQUITY_PATH, index_col="Date", parse_dates=True
        )["BAB_Equity"]
    except Exception as e:
        print(f"   ⚠️  Could not load equity curve: {e}")
        return

    aqr = load_aqr()
    mkt_eq = None
    if aqr is not None and "MKT" in aqr.columns:
        mkt_raw = (1 + aqr["MKT"].dropna()).cumprod()
        mkt_eq  = mkt_raw / mkt_raw.iloc[0]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(bab_eq.index, bab_eq.values, color=COLORS["bab"],
            linewidth=2, label="BAB Strategy", zorder=3)

    if mkt_eq is not None:
        ax.plot(mkt_eq.index, mkt_eq.values, color=COLORS["mkt"],
                linewidth=2, label="Market Portfolio", zorder=2)

    ax.set_yscale("log")
    ax.set_ylabel("Cumulative Wealth (log scale, start = 1.0)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Equity Curve: Hypothetical €1 Invested at Start\n"
                 "(excludes transaction costs)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Annotate final values
    ax.annotate(f"{bab_eq.iloc[-1]:.1f}×",
                xy=(bab_eq.index[-1], bab_eq.iloc[-1]),
                xytext=(-70, 10), textcoords="offset points",
                fontsize=11, color=COLORS["bab"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["bab"]))

    if mkt_eq is not None:
        ax.annotate(f"{mkt_eq.iloc[-1]:.1f}×",
                    xy=(mkt_eq.index[-1], mkt_eq.iloc[-1]),
                    xytext=(-70, -25), textcoords="offset points",
                    fontsize=11, color=COLORS["mkt"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLORS["mkt"]))

    fig.tight_layout()
    save_fig(fig, "chart_3_equity_curve.png")


# ── Chart 4 — Rolling Monthly Returns ────────────────────────

def chart_rolling_returns() -> None:
    """Monthly bar chart + 12-month rolling mean."""
    print("\n   [Chart 4] Rolling returns")

    if not os.path.exists(PORT_PATH):
        print("   ⚠️  Portfolio returns not found — skipping")
        return

    try:
        port = pd.read_csv(
            PORT_PATH, index_col="Date", parse_dates=True
        )["BAB_Return"]
    except Exception as e:
        print(f"   ⚠️  Could not load portfolio returns: {e}")
        return

    rolling_12 = port.rolling(12).mean() * 12   # annualised

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Monthly bar chart — green positive, red negative
    bar_colors = [COLORS["long"] if r >= 0 else COLORS["short"]
                  for r in port]
    ax1.bar(port.index, port.values, color=bar_colors,
            width=25, alpha=0.75, label="Monthly return")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Monthly Return", fontsize=11)
    ax1.set_title("BAB Portfolio Monthly Returns", fontsize=12,
                  fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Rolling 12-month mean
    ax2.plot(rolling_12.index, rolling_12.values, color=COLORS["bab"],
             linewidth=2, label="12-month rolling mean (ann.)")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.fill_between(rolling_12.index, rolling_12.values, 0,
                     where=(rolling_12 >= 0), alpha=0.2,
                     color=COLORS["long"])
    ax2.fill_between(rolling_12.index, rolling_12.values, 0,
                     where=(rolling_12 < 0), alpha=0.2,
                     color=COLORS["short"])
    ax2.set_ylabel("Rolling 12M Return (ann.)", fontsize=11)
    ax2.set_title("12-Month Rolling Mean Return (Annualised)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "chart_4_rolling_returns.png")


# ── Chart 5 — Beta Distribution Histogram ────────────────────

def chart_beta_histogram(screener_df: pd.DataFrame) -> None:
    """Distribution of current S&P 500 betas."""
    print("\n   [Chart 5] Beta histogram")

    if screener_df.empty:
        print("   ⚠️  No screener data — skipping")
        return

    betas  = screener_df["Beta_12M"].dropna()
    low_b  = screener_df.loc[screener_df["Signal"] == "Low",  "Beta_12M"]
    high_b = screener_df.loc[screener_df["Signal"] == "High", "Beta_12M"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(betas,  bins=40, color=COLORS["neutral"],
            alpha=0.5, label="All stocks",        edgecolor="white")
    ax.hist(low_b,  bins=25, color=COLORS["long"],
            alpha=0.75, label="Low-beta (long)",  edgecolor="white")
    ax.hist(high_b, bins=25, color=COLORS["short"],
            alpha=0.75, label="High-beta (short)", edgecolor="white")

    ax.axvline(betas.median(), color="black", linewidth=2,
               linestyle="--",
               label=f"Median β = {betas.median():.2f}")
    ax.axvline(1.0, color="purple", linewidth=1.5,
               linestyle=":", label="β = 1.0 (market)")

    ax.set_xlabel("12-Month Rolling Beta", fontsize=12)
    ax.set_ylabel("Number of Stocks", fontsize=12)
    ax.set_title("S&P 500 Beta Distribution — Current BAB Screener",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "chart_5_beta_histogram.png")


# ── Chart 6 — Top / Bottom Beta Stocks ───────────────────────

def chart_top_bottom_beta(screener_df: pd.DataFrame) -> None:
    """Horizontal bar chart of highest and lowest beta stocks."""
    print("\n   [Chart 6] Top/bottom beta stocks")

    if screener_df.empty:
        print("   ⚠️  No screener data — skipping")
        return

    betas  = screener_df["Beta_12M"].dropna().sort_values()
    top_n  = min(TOP_N, len(betas) // 2)
    low    = betas.head(top_n)
    high   = betas.tail(top_n).iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.barh(low.index, low.values, color=COLORS["long"],
             alpha=0.85, edgecolor="white")
    ax1.axvline(1, color="purple", linewidth=1.2, linestyle=":",
                label="β = 1")
    ax1.set_xlabel("Beta (12-month rolling)", fontsize=11)
    ax1.set_title(f"Top {top_n} Lowest Beta\n(BAB Long Candidates)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh(high.index, high.values, color=COLORS["short"],
             alpha=0.85, edgecolor="white")
    ax2.axvline(1, color="purple", linewidth=1.2, linestyle=":",
                label="β = 1")
    ax2.set_xlabel("Beta (12-month rolling)", fontsize=11)
    ax2.set_title(f"Top {top_n} Highest Beta\n(BAB Short Candidates)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Current S&P 500 — BAB Screener Rankings",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "chart_6_top_bottom_beta.png")


# ── Chart 7 — Correlation Heatmap ────────────────────────────

def chart_correlation_heatmap() -> None:
    """Correlation matrix of BAB, MKT, SMB, HML factors."""
    print("\n   [Chart 7] Correlation heatmap")

    aqr = load_aqr()
    if aqr is None:
        return

    # Select factor columns that exist
    factor_cols = [c for c in ["BAB", "MKT", "SMB", "HML"] if c in aqr.columns]
    if len(factor_cols) < 2:
        print("   ⚠️  Not enough factor columns — skipping heatmap")
        return

    corr_matrix = aqr[factor_cols].dropna().corr()

    fig, ax = plt.subplots(figsize=(8, 6))

    if HAS_SEABORN:
        sns.heatmap(
            corr_matrix,
            annot      = True,
            fmt        = ".3f",
            cmap       = "RdBu_r",
            center     = 0,
            vmin       = -1, vmax = 1,
            linewidths = 0.5,
            ax         = ax,
        )
    else:
        # Fallback: matplotlib imshow
        im = ax.imshow(corr_matrix.values, cmap="RdBu_r",
                       vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(factor_cols)))
        ax.set_yticks(range(len(factor_cols)))
        ax.set_xticklabels(factor_cols)
        ax.set_yticklabels(factor_cols)

        # For loop to annotate each cell
        for i in range(len(factor_cols)):
            for j in range(len(factor_cols)):
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.3f}",
                        ha="center", va="center", fontsize=11)

    ax.set_title("Factor Correlation Matrix\n(BAB, MKT, SMB, HML)",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    save_fig(fig, "chart_7_correlation_heatmap.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("PERSON 5 — BAB SCREENER + ALL VISUALIZATIONS")
    print("=" * 60)

    ensure_dirs()

    # ── Part A: Live Screener ─────────────────────────────────
    screener_df = run_screener()

    # Google Sheets export (bonus library)
    export_to_google_sheets(screener_df)

    # ── Part B: All Charts ────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART B: Generating all charts")
    print("=" * 60)

    # List of chart functions to call in a for loop
    chart_functions = [
        chart_boxplots,
        chart_qqplots,
        chart_equity_curve,
        chart_rolling_returns,
        lambda: chart_beta_histogram(screener_df),
        lambda: chart_top_bottom_beta(screener_df),
        chart_correlation_heatmap,
    ]

    # For loop over all chart functions
    for chart_fn in chart_functions:
        try:
            chart_fn()
        except Exception as e:
            print(f"   ❌ Chart error: {e}")

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print(f"   Screener → {SCREENER_OUT}")
    print(f"   Charts   → {FIGURES_DIR}/  (7 charts)")
    print("=" * 60)

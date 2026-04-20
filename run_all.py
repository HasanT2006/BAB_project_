# ============================================================
# run_all.py — Master Runner
# Run this from the project root to execute all 5 persons' scripts.
#
#   python run_all.py
# ============================================================

import subprocess
import sys
import os
import time

# Ordered list of (person label, script path)
PIPELINE = [
    ("Person 1 — AQR Data Loader",           "code1_backtest/person1_data_loader.py"),
    ("Person 1 — Fetch S&P 500 Data",        "code2_screener/person1_fetch_current.py"),
    ("Person 3 — BAB Portfolio Backtest",     "code1_backtest/person3_bab_portfolio.py"),
    ("Person 4 — Statistical Analysis",       "code1_backtest/person4_statistical_analysis.py"),
    ("Person 5 — Screener + Visualizations",  "code2_screener/person5_screener_and_charts.py"),
]


def run_step(label, script):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Script: {script}")
    print(f"{'='*60}")

    if not os.path.exists(script):
        print(f"  ❌ Script not found — skipping")
        return False

    start  = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✅ Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ❌ FAILED (exit code {result.returncode})")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BAB STRATEGY — FULL PIPELINE")
    print("=" * 60)

    summary = []
    for label, script in PIPELINE:
        ok = run_step(label, script)
        summary.append((label, "✅ OK" if ok else "❌ FAILED"))

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    for label, status in summary:
        print(f"  {status}  {label}")
    print("=" * 60)

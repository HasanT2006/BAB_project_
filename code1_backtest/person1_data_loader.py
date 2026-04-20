# ============================================================
# PERSON 1 — AQR Data Loader
# File: code1_backtest/person1_data_loader.py
#
# ROLE IN PROJECT:
#   Loads the AQR Betting-Against-Beta Excel dataset,
#   extracts European factor data, cleans it, and saves
#   a ready-to-use CSV for the whole team.
#
# MARKS TARGETED:
#   Basic Programming (30%):
#     - Dictionary: SHEETS maps sheet names → column names
#     - List: all_series collects loaded Series
#     - For Loop: iterates over every sheet
#     - Functions: each step is its own function
#     - Error Handling: try/except on every file operation
#     - Conditions: cache check, missing column fallback
#   Libraries (40%):
#     - pandas: read_excel, DataFrame, date manipulation
#     - openpyxl: backend for reading .xlsx files
#
# INPUT:
#   data/raw/aqr_bab_data.xlsx
#   Download from:
#   https://www.aqr.com/Insights/Datasets/
#   Betting-Against-Beta-Equity-Factors-Monthly
#
# OUTPUT:
#   data/processed/aqr_returns.csv
#   Columns: BAB, MKT, SMB, HML, RF
#
# HOW TO RUN:
#   python code1_backtest/person1_data_loader.py
# ============================================================

import pandas as pd
import os
import sys

# ============================================================
# SETTINGS  (Dictionary — basic programming mark)
# ============================================================

RAW_FILE_PATH = "data/raw/aqr_bab_data.xlsx"
OUTPUT_PATH   = "data/processed/aqr_returns.csv"

# Dictionary: AQR sheet name → our column name
SHEETS = {
    "BAB Factors" : "BAB",
    "MKT"         : "MKT",
    "SMB"         : "SMB",
    "HML FF"      : "HML",
}

RF_SHEET   = "RF"
EUR_COLUMN = "Europe"   # Column we want from each sheet

START_DATE = "1990-01-01"
END_DATE   = "2023-05-31"

PREVIEW_ROWS = 5


# ============================================================
# STEP 1 — CHECK FILE EXISTS
# ============================================================

def check_file_exists(filepath):
    """
    Checks whether the AQR Excel file is present.
    Returns True if found, False otherwise.
    Uses: condition (if/else), error handling
    """
    print("\n" + "=" * 55)
    print("STEP 1: Checking file exists")
    print("=" * 55)

    # Condition: does file exist?
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   ✅ File found : {filepath}")
        print(f"      Size       : {size_mb:.2f} MB")
        return True
    else:
        print(f"   ❌ File not found : {filepath}")
        print(f"   Please download from AQR and save to data/raw/")
        print(f"   URL: https://www.aqr.com/Insights/Datasets/"
              f"Betting-Against-Beta-Equity-Factors-Monthly")
        return False


# ============================================================
# STEP 2 — LOAD ONE FACTOR SHEET
# ============================================================

def load_factor_sheet(sheet_name, column_name):
    """
    Loads a single AQR factor sheet and extracts the Europe column.

    Parameters:
        sheet_name  : str — AQR Excel sheet name
        column_name : str — what to call the column in our output

    Returns:
        pd.Series or None on failure

    Uses: try/except (error handling), condition (column fallback)
    """
    try:
        # Read the sheet — AQR has 18 rows of description before data
        df = pd.read_excel(
            RAW_FILE_PATH,
            sheet_name  = sheet_name,
            skiprows    = 18,
            index_col   = 0,
            parse_dates = True
        )

        # Condition: try Europe column, fall back if not found
        if EUR_COLUMN in df.columns:
            col = EUR_COLUMN

        else:
            # List comprehension to find any Europe-like column
            candidates = [c for c in df.columns if "eur" in c.lower()]

            if candidates:
                col = candidates[0]
                print(f"   ℹ️  Using '{col}' for {sheet_name}")
            elif "USA" in df.columns:
                col = "USA"
                print(f"   ⚠️  Europe not found in {sheet_name} — falling back to USA")
            else:
                print(f"   ❌ No suitable column in {sheet_name}")
                print(f"      Available: {list(df.columns[:5])}")
                return None

        series = df[col].rename(column_name)
        print(f"   ✅ {sheet_name:<15} : {len(series)} rows loaded")
        return series

    except Exception as e:
        print(f"   ❌ ERROR loading {sheet_name}: {e}")
        return None


# ============================================================
# STEP 3 — LOAD RF SHEET (different structure)
# ============================================================

def load_rf_sheet():
    """
    Loads the Risk-Free Rate sheet.
    AQR RF sheet only has two columns: DATE + Rate.
    Uses: try/except
    """
    try:
        df = pd.read_excel(
            RAW_FILE_PATH,
            sheet_name  = RF_SHEET,
            skiprows    = 18,
            index_col   = 0,
            parse_dates = True
        )
        df.columns = ["RF"]
        print(f"   ✅ {'RF':<15} : {len(df)} rows loaded")
        return df["RF"]

    except Exception as e:
        print(f"   ❌ ERROR loading RF sheet: {e}")
        return None


# ============================================================
# STEP 4 — LOAD AND COMBINE ALL SHEETS
# ============================================================

def load_all_sheets():
    """
    Loops over the SHEETS dictionary and combines everything.
    Uses: for loop, list (all_series), dictionary iteration
    """
    print("\n" + "=" * 55)
    print("STEP 2: Loading all sheets")
    print("=" * 55)

    # List to collect all Series before combining
    all_series = []

    # For loop over dictionary items
    for sheet_name, column_name in SHEETS.items():
        series = load_factor_sheet(sheet_name, column_name)

        # Condition: only add if loading succeeded
        if series is not None:
            all_series.append(series)

    # Load RF separately
    rf_series = load_rf_sheet()
    if rf_series is not None:
        all_series.append(rf_series)

    # Error check: did we get all 5 columns?
    if len(all_series) < 5:
        print(f"\n   ❌ Expected 5 columns, got {len(all_series)}")
        return None

    # Combine list of Series into one DataFrame
    df = pd.concat(all_series, axis=1)
    print(f"\n   Combined shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns        : {list(df.columns)}")
    return df


# ============================================================
# STEP 5 — CLEAN DATA
# ============================================================

def clean_data(df):
    """
    Normalises dates, filters to study period, removes blanks.
    Uses: conditions, pandas date methods
    """
    print("\n" + "=" * 55)
    print("STEP 3: Cleaning data")
    print("=" * 55)

    # Normalise all dates to month-end so team code aligns correctly
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("M").to_timestamp("M")
    df.index.name = "Date"
    df = df.sort_index()

    # Filter to study period
    df = df.loc[START_DATE:END_DATE]
    print(f"   Date range : {START_DATE} → {END_DATE}")
    print(f"   Rows kept  : {len(df)}")

    # Drop rows where every column is NaN
    before = len(df)
    df = df.dropna(how="all")
    dropped = before - len(df)
    if dropped > 0:
        print(f"   Dropped {dropped} fully empty rows")

    # Convert all columns to numeric (AQR sometimes stores as object)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Report missing values honestly
    missing_counts = df.isnull().sum()   # Series: column → count
    total_missing  = missing_counts.sum()

    if total_missing == 0:
        print(f"   ✅ No missing values")
    else:
        print(f"   ⚠️  Missing values found:")
        # For loop over missing value dictionary
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"      {col}: {count} missing months")

    print(f"   Final shape : {df.shape[0]} months × {df.shape[1]} columns")
    return df


# ============================================================
# STEP 6 — PREVIEW
# ============================================================

def preview_data(df):
    """Prints a readable summary of the loaded data."""
    print("\n" + "=" * 55)
    print("STEP 4: Data Preview")
    print("=" * 55)

    print(f"\nFirst {PREVIEW_ROWS} rows:")
    print(df.head(PREVIEW_ROWS).round(6).to_string())

    print(f"\nLast {PREVIEW_ROWS} rows:")
    print(df.tail(PREVIEW_ROWS).round(6).to_string())

    print(f"\nBasic statistics:")
    print(df.describe().round(4).to_string())

    # Dictionary for column descriptions
    descriptions = {
        "BAB" : "Betting-Against-Beta return (Europe)",
        "MKT" : "Market excess return (Europe)",
        "SMB" : "Small Minus Big factor (Europe)",
        "HML" : "High Minus Low factor (Europe, Fama-French)",
        "RF"  : "Risk-free rate (monthly)",
    }

    print(f"\nColumn meanings:")
    for col, desc in descriptions.items():
        print(f"   {col} = {desc}")


# ============================================================
# STEP 7 — SAVE
# ============================================================

def save_data(df):
    """
    Saves the clean DataFrame to CSV.
    Uses: try/except, os.makedirs
    """
    print("\n" + "=" * 55)
    print("STEP 5: Saving data")
    print("=" * 55)

    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH)
        size_kb = os.path.getsize(OUTPUT_PATH) / 1024
        print(f"   ✅ Saved : {OUTPUT_PATH}  ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"   ❌ ERROR saving: {e}")
        return False


# ============================================================
# CACHE CHECK
# ============================================================

def load_from_cache():
    """
    If processed file already exists, skip re-processing.
    Delete data/processed/aqr_returns.csv to force fresh load.
    Uses: condition, try/except
    """
    if os.path.exists(OUTPUT_PATH):
        print(f"\n   ✅ Cache found: {OUTPUT_PATH}")
        print(f"      Delete to force fresh load.")
        try:
            df = pd.read_csv(OUTPUT_PATH, index_col="Date", parse_dates=True)
            print(f"      Loaded : {df.shape[0]} months × {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"   ⚠️  Cache load failed ({e}) — re-processing...")
            return None
    return None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("PERSON 1 — AQR DATA LOADER")
    print("=" * 55)

    # Check cache first
    cached = load_from_cache()
    if cached is not None:
        preview_data(cached)
        sys.exit(0)

    # Step 1: File check
    if not check_file_exists(RAW_FILE_PATH):
        sys.exit(1)

    # Step 2: Load all sheets
    df_raw = load_all_sheets()
    if df_raw is None:
        sys.exit(1)

    # Step 3: Clean
    df_clean = clean_data(df_raw)

    # Step 4: Preview
    preview_data(df_clean)

    # Step 5: Save
    saved = save_data(df_clean)

    print("\n" + "=" * 55)
    print("FOR YOUR TEAMMATES")
    print("=" * 55)
    if saved:
        print(f"   Load the data like this:")
        print(f"   df = pd.read_csv('{OUTPUT_PATH}',")
        print(f"                    index_col='Date', parse_dates=True)")
        print(f"   Columns: BAB, MKT, SMB, HML, RF")
    else:
        print("   ❌ Script failed — check errors above")
    print("=" * 55)

"""
output_x.py
===========
Prepares feature matrices (X, y) for all model specifications.

Four datasets
-------------
  X1 — Simple average
        Monthly variables averaged within each quarter, joined with quarterly data.
        Shape: (n_quarters, n_monthly + n_quarterly)

  X2 — Simple average + lags
        Same as X1 (call it qd1), then add 4 quarterly lags of every column in qd1.
        Shape: (n_quarters, (n_monthly + n_quarterly) × 5)

  X3 — U-MIDAS
        Monthly variables kept as 3 separate features per quarter (_m1/_m2/_m3),
        joined with quarterly data.
        Shape: (n_quarters, n_monthly×3 + n_quarterly)

  X4 — U-MIDAS + lags
        U-MIDAS monthly block (current quarter) + 4 quarterly lags of that block
        (= 12 monthly observations per variable), plus quarterly data + 4 lags.
        Shape: (n_quarters, n_monthly×3×5 + n_quarterly×5)
"""


import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

THIS_DIR     = Path(__file__).resolve().parent   # pipeline/
PROJECT_DIR  = THIS_DIR.parent                   # project root

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_DIR))
from ragged_edge import read_table
from database.client import get_backend_client


# =============================================================================
# DATA LOADING
# =============================================================================

def load_filled_data():
    """Fetch filled_md and filled_qd from Supabase."""
    print("Loading filled data from Supabase …")
    supabase = get_backend_client()

    df_md = read_table(supabase, "filled_md")
    df_md["sasdate"] = pd.to_datetime(df_md["sasdate"])
    df_md = df_md.sort_values("sasdate").reset_index(drop=True)
    print(f"  filled_md : {df_md.shape}")

    df_qd = read_table(supabase, "filled_qd")
    df_qd["sasdate"] = pd.to_datetime(df_qd["sasdate"])
    df_qd = df_qd.sort_values("sasdate").reset_index(drop=True)
    print(f"  filled_qd : {df_qd.shape}\n")

    return df_md, df_qd


def _load_gdp():
    supabase = get_backend_client()
    gdp = read_table(supabase, "gdp")
    gdp["sasdate"] = pd.to_datetime(gdp["sasdate"])
    gdp = gdp.set_index("sasdate").sort_index()
    gdp = gdp[gdp.index.notna()]
    return gdp


def _load_gdp_with_flash() -> pd.Series:
    """
    Returns the GDP growth series (GDPC1_t) with any unreleased quarters
    filled by the latest Ensemble flash prediction from model_forecasts.

    Edge case: on 31 March 2026, Q4 2025 GDP may not yet be officially
    released. We substitute the Ensemble nowcast so lag features can be
    constructed for the Q1 2026 nowcast row.
    """
    gdp = _load_gdp()
    y = gdp["GDPC1_t"].copy()

    missing = y[y.isna()].index
    if len(missing) == 0:
        return y

    supabase = get_backend_client()
    for date in missing:
        resp = (supabase.table("model_forecasts")
                .select("nowcast")
                .eq("model_name", "All_Model_Average")
                .eq("quarter_date", date.strftime("%Y-%m-%d"))
                .order("month_date", desc=True)
                .execute())
        if resp.data:
            flash_val = float(resp.data[0]["nowcast"])
            y[date] = flash_val
            print(f"  GDP lag: using Ensemble flash prediction "
                  f"{flash_val:.4f} for {date.date()} (not yet officially released)")
        else:
            print(f"  GDP lag: no flash prediction found for {date.date()}, "
                  f"lag will remain NaN")
    return y


def _load_gdp_lags(n_lags: int = 4) -> pd.DataFrame:
    """
    Returns a DataFrame of n_lags quarterly lags of GDP growth.
    Columns: gdp_lag1 … gdp_lag{n_lags}.

    Uses _load_gdp_with_flash() so that unreleased quarters (e.g. Q4 2025
    on 31 March 2026) are filled by the Ensemble flash prediction before
    computing lags, ensuring the t row (e.g. Q1 2026) has valid lag features.
    """
    y = _load_gdp_with_flash()
    df = y.to_frame()
    for k in range(1, n_lags + 1):
        df[f"gdp_lag{k}"] = y.shift(k)
    lag_cols = [f"gdp_lag{k}" for k in range(1, n_lags + 1)]
    return df[lag_cols]


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _average_monthly_to_quarterly(df_md: pd.DataFrame) -> pd.DataFrame:
    """
    Average the 3 monthly observations within each quarter per variable.
    Excludes COVID flag columns (kept as quarterly-level features in df_qd).
    Returns DataFrame indexed by quarter label, columns suffixed with _md.
    """
    df = df_md.copy().sort_values("sasdate").reset_index(drop=True)
    feature_cols = [c for c in df.columns
                    if c not in ("sasdate", "covid_crash", "covid_recover")]
    df["qtr_label"] = (
        df["sasdate"].dt.to_period("Q").dt.start_time
        + pd.DateOffset(months=2)
    )
    df_agg = (
        df.groupby("qtr_label")[feature_cols]
        .mean()
        .rename_axis("sasdate")
        .add_suffix("_md")
    )
    return df_agg


def _umidas_monthly_to_quarterly(df_md: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly data to quarterly U-MIDAS features.
    Each variable becomes 3 columns: _m1 (oldest), _m2, _m3 (most recent).
    Excludes COVID flag columns.
    Returns DataFrame indexed by quarter label.
    """
    df = df_md.copy().sort_values("sasdate").reset_index(drop=True)
    df["qtr_label"] = (
        df["sasdate"].dt.to_period("Q").dt.start_time
        + pd.DateOffset(months=2)
    )
    df["month_pos"] = df.groupby("qtr_label").cumcount() + 1
    feature_cols = [c for c in df.columns
                    if c not in ("sasdate", "qtr_label", "month_pos",
                                 "covid_crash", "covid_recover")]
    df_pivot = df.pivot(index="qtr_label", columns="month_pos", values=feature_cols)
    df_pivot.columns = [f"{col}_m{pos}" for col, pos in df_pivot.columns]
    df_pivot.index.name = "sasdate"
    return df_pivot


def _prep_qd(df_qd: pd.DataFrame) -> pd.DataFrame:
    df = df_qd.copy()
    df["sasdate"] = pd.to_datetime(df["sasdate"])
    return df.set_index("sasdate").sort_index()


def _add_lags(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """
    Add n_lags quarterly lags of every column in df.
    Lag-k columns are named {col}_lag{k}.
    The lag-0 (current) columns keep their original names.
    """
    lagged = [df]
    for k in range(1, n_lags + 1):
        shifted = df.shift(k).add_suffix(f"_lag{k}")
        lagged.append(shifted)
    return pd.concat(lagged, axis=1)


def _finalise(X: pd.DataFrame, gdp: pd.DataFrame) -> tuple:
    """
    Align y to X's index, drop NaN rows, return (X, y).

    Rows where X is complete but y is NaN are retained only if they fall
    after the last known GDP date — these are the nowcast rows (e.g. 2026 Q1).
    Rows before the GDP series begins are dropped.

    NOTE: y will be NaN for nowcast rows. When passing to poos_validation(),
    use only rows where y.notna() for evaluation, and use the last row of X
    separately for the actual nowcast prediction.
    """
    X = X.reindex(gdp.index)
    y = gdp["GDPC1_t"]
    valid = X.notna().all(axis=1) 
    if (~valid).sum() > 0:
        print(f"  Dropping {(~valid).sum()} rows with NaNs.")
    return X[valid], y[valid]



# =============================================================================
# X1 — SIMPLE AVERAGE
# =============================================================================

def build_X1(df_md: pd.DataFrame, df_qd: pd.DataFrame) -> tuple:
    """
    X1: averaged monthly features + quarterly features.

    Monthly variables are averaged across the 3 months within each quarter.
    """
    gdp   = _load_gdp()
    df_avg = _average_monthly_to_quarterly(df_md)
    df_q   = _prep_qd(df_qd)
    X = df_avg.join(df_q, how="inner").join(_load_gdp_lags(), how="left")
    X, y = _finalise(X, gdp)
    print(f"X1 (avg):            {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X2 — SIMPLE AVERAGE + LAGS
# =============================================================================

def build_X2(df_md: pd.DataFrame, df_qd: pd.DataFrame, n_lags: int = 4) -> tuple:
    """
    X2: averaged monthly + quarterly (call it qd1), then add n_lags quarterly
    lags of every column in qd1.

    Total features = (n_monthly_avg + n_quarterly) × (1 + n_lags)
    """
    gdp   = _load_gdp()
    df_avg = _average_monthly_to_quarterly(df_md)
    df_q   = _prep_qd(df_qd)
    qd1 = df_avg.join(df_q, how="inner")
    X = _add_lags(qd1, n_lags).join(_load_gdp_lags(), how="left")
    X, y = _finalise(X, gdp)
    print(f"X2 (avg + {n_lags} lags):     {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X3 — U-MIDAS
# =============================================================================

def build_X3(df_md: pd.DataFrame, df_qd: pd.DataFrame) -> tuple:
    """
    X3: U-MIDAS monthly features (_m1/_m2/_m3) + quarterly features.

    Each monthly variable becomes 3 quarterly features preserving
    within-quarter dynamics.
    """
    gdp      = _load_gdp()
    df_umidas = _umidas_monthly_to_quarterly(df_md)
    df_q      = _prep_qd(df_qd)
    X = df_umidas.join(df_q, how="inner").join(_load_gdp_lags(), how="left")
    X, y = _finalise(X, gdp)
    print(f"X3 (U-MIDAS):        {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X4 — U-MIDAS + LAGS
# =============================================================================

def build_X4(df_md: pd.DataFrame, df_qd: pd.DataFrame,
             n_monthly_lags: int = 4, n_qd_lags: int = 4) -> tuple:
    """
    X4: U-MIDAS monthly block + n_monthly_lags quarterly lags of that block,
    plus quarterly data + n_qd_lags quarterly lags.

    n_monthly_lags=4 means 4 quarterly shifts of the _m1/_m2/_m3 block,
    covering 4×3=12 monthly observations of history per variable.

    n_qd_lags=4 means 4 quarterly lags of each quarterly variable.
    """
    gdp       = _load_gdp()
    df_umidas = _umidas_monthly_to_quarterly(df_md)
    df_q      = _prep_qd(df_qd)

    df_umidas_lagged = _add_lags(df_umidas, n_monthly_lags)
    df_q_lagged      = _add_lags(df_q, n_qd_lags)

    X = df_umidas_lagged.join(df_q_lagged, how="inner").join(_load_gdp_lags(), how="left")
    X, y = _finalise(X, gdp)
    print(f"X4 (U-MIDAS + lags): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X_AR — AUTOREGRESSIVE BENCHMARK (2 GDP lags)
# =============================================================================

def build_X_AR(n_lags: int = 2) -> tuple:
    """
    X_AR: minimal GDP-only AR benchmark, direct-forecast on real lags only.

    Unlike the other builders, this does NOT use _load_gdp_with_flash() —
    the benchmark must not depend on the ensemble's own predictions. If
    GDP at t-1 is released, uses a standard 1-step AR(2) (lag_a=t-1,
    lag_b=t-2). If t-1 is not yet released, uses a direct 2-step forecast
    (lag_a=t-2, lag_b=t-3) instead of imputing the missing lag.
    """
    y_gdp = _load_gdp()["GDPC1_t"]
    df = y_gdp.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 2):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)

    t1_available = pd.notna(df["lag_1"].iloc[-1])
    lag_cols = ["lag_1", "lag_2"] if t1_available else ["lag_2", "lag_3"]

    df_hist = df.iloc[:-1][["gdp_growth"] + lag_cols].dropna()
    X_hist = df_hist[lag_cols].rename(columns={lag_cols[0]: "lag_a", lag_cols[1]: "lag_b"})
    y_hist = df_hist["gdp_growth"]

    last_row = df.iloc[[-1]]
    X_last = last_row[lag_cols].rename(columns={lag_cols[0]: "lag_a", lag_cols[1]: "lag_b"})
    y_last = last_row["gdp_growth"]

    X = pd.concat([X_hist, X_last])
    y = pd.concat([y_hist, y_last])
    print(f"X_AR (direct-forecast, using {lag_cols}): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    DATA_DIR = PROJECT_DIR / "data"

    df_md, df_qd = load_filled_data()

    X1, y1             = build_X1(df_md, df_qd)
    X2, y2             = build_X2(df_md, df_qd, n_lags=4)
    X3, y3             = build_X3(df_md, df_qd)
    X4, y4             = build_X4(df_md, df_qd, n_monthly_lags=4, n_qd_lags=4)
    X_AR, y_AR         = build_X_AR(n_lags=2)

    datasets = [
        ("X1",       X1,       y1),
        ("X2",       X2,       y2),
        ("X3",       X3,       y3),
        ("X4",       X4,       y4),
        ("X_AR",     X_AR,     y_AR),
    ]

    print("\n=== Saving to CSV ===")
    for name, X, y in datasets:
        x_path = DATA_DIR / f"{name}.csv"
        y_path = DATA_DIR / f"y_{name}.csv"
        X.to_csv(x_path)
        y.to_csv(y_path, header=True)
        print(f"  {name}: X={X.shape} → {x_path.name}, y={y.shape} → {y_path.name}")

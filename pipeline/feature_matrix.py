"""
feature_matrix.py
==================
Prepares feature matrices (X, y) for all model specifications, for both the
live nowcasting path and POOS (pseudo-out-of-sample) evaluation.

Every builder is a pure function: no Supabase access, no filesystem access.
Callers (pipeline.prediction, pipeline.poos) are responsible for assembling
the two GDP series each builder needs — see pipeline.gdp_data for the
live-path loaders, and pipeline.poos.cut_and_fill for the POOS-path cuts.

  gdp_for_lags — feeds the gdp_lag* feature columns. Flash-filled for
                 X1-X4 (an unreleased quarter's lag can use the ensemble's
                 own flash prediction); raw/point-in-time-cut for the AR
                 benchmarks, which must never depend on the ensemble.
  gdp_actual   — the raw series used only to look up y (never flash-filled).

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

import pandas as pd
from dateutil.relativedelta import relativedelta

# Real GDP components (consumption, investment, exports) that would let a
# model near-reconstruct GDP if used contemporaneously. Kept for their
# predictive lag information, but never as same-quarter regressors.
GDP_COMPONENT_COLS = ["PCECC96_t_qd", "GPDIC1_t_qd", "EXPGSC1_t_qd"]


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


def _build_gdp_lags(gdp_for_lags: pd.Series, index: pd.DatetimeIndex, n_lags: int = 4) -> pd.DataFrame:
    """Builds GDP lag columns (gdp_lag1..gdp_lag{n_lags}) aligned to index."""
    reindexed = gdp_for_lags.reindex(index)
    out = pd.DataFrame(index=index)
    for k in range(1, n_lags + 1):
        out[f"gdp_lag{k}"] = reindexed.shift(k)
    return out


def _finalise(X: pd.DataFrame, gdp_actual: pd.Series) -> tuple:
    """
    Align y to X's index, drop any row where X has an incomplete feature.

    Rows where X is complete but y is NaN are retained only if they fall
    after the last known GDP date — these are the nowcast rows (e.g. 2026 Q1).
    Rows before the GDP series begins are dropped.

    NOTE: y will be NaN for nowcast rows. When passing to poos_validation(),
    use only rows where y.notna() for evaluation, and use the last row of X
    separately for the actual nowcast prediction.
    """
    X = X.reindex(gdp_actual.index)
    y = gdp_actual
    valid = X.notna().all(axis=1)
    if (~valid).sum() > 0:
        print(f"  Dropping {(~valid).sum()} rows with NaNs.")
    return X[valid], y[valid]


# =============================================================================
# X1 — SIMPLE AVERAGE
# =============================================================================

def build_X1(df_md: pd.DataFrame, df_qd: pd.DataFrame,
             gdp_for_lags: pd.Series, gdp_actual: pd.Series) -> tuple:
    """
    X1: averaged monthly features + quarterly features.

    Monthly variables are averaged across the 3 months within each quarter.
    """
    df_avg = _average_monthly_to_quarterly(df_md)
    df_q   = _prep_qd(df_qd)
    missing = [c for c in GDP_COMPONENT_COLS if c not in df_q.columns]
    if missing:
        raise ValueError(f"GDP component columns not found in quarterly data: {missing}")
    df_q = df_q.drop(columns=GDP_COMPONENT_COLS)
    X = df_avg.join(df_q, how="inner")
    X = X.join(_build_gdp_lags(gdp_for_lags, X.index), how="left")
    X, y = _finalise(X, gdp_actual)
    print(f"X1 (avg):            {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X2 — SIMPLE AVERAGE + LAGS
# =============================================================================

def build_X2(df_md: pd.DataFrame, df_qd: pd.DataFrame,
             gdp_for_lags: pd.Series, gdp_actual: pd.Series, n_lags: int = 4) -> tuple:
    """
    X2: averaged monthly + quarterly (call it qd1), then add n_lags quarterly
    lags of every column in qd1.

    Total features = (n_monthly_avg + n_quarterly) × (1 + n_lags)
    """
    df_avg = _average_monthly_to_quarterly(df_md)
    df_q   = _prep_qd(df_qd)
    qd1 = df_avg.join(df_q, how="inner")
    X = _add_lags(qd1, n_lags).drop(columns=GDP_COMPONENT_COLS, errors="ignore")
    X = X.join(_build_gdp_lags(gdp_for_lags, X.index), how="left")
    X, y = _finalise(X, gdp_actual)
    print(f"X2 (avg + {n_lags} lags):     {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X3 — U-MIDAS
# =============================================================================

def build_X3(df_md: pd.DataFrame, df_qd: pd.DataFrame,
             gdp_for_lags: pd.Series, gdp_actual: pd.Series) -> tuple:
    """
    X3: U-MIDAS monthly features (_m1/_m2/_m3) + quarterly features.

    Each monthly variable becomes 3 quarterly features preserving
    within-quarter dynamics.
    """
    df_umidas = _umidas_monthly_to_quarterly(df_md)
    df_q      = _prep_qd(df_qd)
    missing = [c for c in GDP_COMPONENT_COLS if c not in df_q.columns]
    if missing:
        raise ValueError(f"GDP component columns not found in quarterly data: {missing}")
    df_q = df_q.drop(columns=GDP_COMPONENT_COLS)
    X = df_umidas.join(df_q, how="inner")
    X = X.join(_build_gdp_lags(gdp_for_lags, X.index), how="left")
    X, y = _finalise(X, gdp_actual)
    print(f"X3 (U-MIDAS):        {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X4 — U-MIDAS + LAGS
# =============================================================================

def build_X4(df_md: pd.DataFrame, df_qd: pd.DataFrame,
             gdp_for_lags: pd.Series, gdp_actual: pd.Series,
             n_monthly_lags: int = 4, n_qd_lags: int = 4, n_gdp_lags: int = 4) -> tuple:
    """
    X4: U-MIDAS monthly block + n_monthly_lags quarterly lags of that block,
    plus quarterly data + n_qd_lags quarterly lags.

    n_monthly_lags=4 means 4 quarterly shifts of the _m1/_m2/_m3 block,
    covering 4×3=12 monthly observations of history per variable.

    n_qd_lags=4 means 4 quarterly lags of each quarterly variable.

    Uses an INNER join between the U-MIDAS monthly block and the quarterly
    block (matching the live nowcast path's historical behavior) — a row
    is only kept if every feature is present. A prior POOS-only version of
    this function used a left join intended to "keep all U-MIDAS quarters,"
    but that interacted badly with downstream POOS windowing logic that
    discards an entire evaluation quarter if any NaN survives anywhere in
    its training window: one missing raw value could silently poison up to
    TRAIN_SIZE quarters' worth of unrelated evaluations. Inner join instead
    drops exactly the row(s) that are actually incomplete.
    """
    df_umidas = _umidas_monthly_to_quarterly(df_md)
    df_q      = _prep_qd(df_qd)

    df_umidas_lagged = _add_lags(df_umidas, n_monthly_lags)
    df_q_lagged      = _add_lags(df_q, n_qd_lags).drop(columns=GDP_COMPONENT_COLS, errors="ignore")

    X = df_umidas_lagged.join(df_q_lagged, how="inner")
    X = X.join(_build_gdp_lags(gdp_for_lags, X.index, n_gdp_lags), how="left")
    X, y = _finalise(X, gdp_actual)
    print(f"X4 (U-MIDAS + lags): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X_AR — AUTOREGRESSIVE BENCHMARK (2 GDP lags)
# =============================================================================

def build_X_AR(gdp_for_lags: pd.Series, gdp_actual: pd.Series, n_lags: int = 2) -> tuple:
    """
    X_AR: minimal GDP-only AR benchmark, direct-forecast on real lags only.

    Live-path version: assumes gdp_for_lags' own last row already IS the
    target/nowcast quarter (its value may be NaN if not yet released) —
    this matches how the Supabase gdp table is maintained (a placeholder
    row exists for the current quarter before official release). For the
    POOS-path equivalent, which must synthesize that target row itself
    from a deliberately-truncated historical cut, see build_X_AR_from_cut.

    Never uses a flash-filled series — the benchmark must not depend on
    the ensemble's own predictions. If GDP at t-1 is released, uses a
    standard 1-step AR(2) (lag_a=t-1, lag_b=t-2). If t-1 is not yet
    released, uses a direct 2-step forecast (lag_a=t-2, lag_b=t-3) instead
    of imputing the missing lag.
    """
    df = gdp_for_lags.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 2):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)

    t1_available = pd.notna(df["lag_1"].iloc[-1])
    lag_cols = ["lag_1", "lag_2"] if t1_available else ["lag_2", "lag_3"]

    df_hist = df.iloc[:-1][["gdp_growth"] + lag_cols]
    df_hist = df_hist[df_hist[lag_cols].notna().all(axis=1)]
    X_hist = df_hist[lag_cols].rename(columns={lag_cols[0]: "lag_a", lag_cols[1]: "lag_b"})
    y_hist = df_hist["gdp_growth"]

    last_row = df.iloc[[-1]]
    X_last = last_row[lag_cols].rename(columns={lag_cols[0]: "lag_a", lag_cols[1]: "lag_b"})
    y_last = last_row["gdp_growth"]

    X = pd.concat([X_hist, X_last])
    y = pd.concat([y_hist, y_last])
    print(f"X_AR (direct-forecast, using {lag_cols}): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


def build_X_AR_from_cut(gdp_for_lags: pd.Series, gdp_actual: pd.Series, n_lags: int = 2) -> tuple:
    """
    X_AR: minimal GDP-only AR benchmark, direct-forecast on real lags only.

    POOS-path version: gdp_for_lags must be the RAW (un-imputed) GDP series
    cut at the information set available at prediction time — no
    ensemble/mean fill-in — and, unlike build_X_AR, does NOT already
    contain a row for the target quarter (it's a historical vintage that
    stops one quarter short). This function synthesizes that target row
    by extending the index by one quarter before applying the same
    t-1-availability fallback as build_X_AR. See build_X_AR's docstring
    for why these two are not merged into one function.
    """
    target_date = gdp_for_lags.index[-1] + relativedelta(months=3)
    full_index  = gdp_for_lags.index.append(pd.DatetimeIndex([target_date]))
    gdp_reindexed = gdp_for_lags.reindex(full_index)

    df = gdp_reindexed.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 2):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)

    t1_available = pd.notna(df["lag_1"].iloc[-1])
    lag_cols = ["lag_1", "lag_2"] if t1_available else ["lag_2", "lag_3"]

    df = df[df[lag_cols].notna().all(axis=1)]
    X = df[lag_cols].rename(columns={lag_cols[0]: "lag_a", lag_cols[1]: "lag_b"})
    y = gdp_actual.reindex(X.index)
    print(f"X_AR (direct-forecast, using {lag_cols}): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# MAIN — live-path CSV export, mirrors the old live-nowcast CSV-export script
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_DIR / "data"

    df_md, df_qd = load_filled_data()
    gdp_actual_series = load_gdp()["GDPC1_t"]
    gdp_flash_series  = load_gdp_with_flash()

    X1, y1     = build_X1(df_md, df_qd, gdp_flash_series, gdp_actual_series)
    X2, y2     = build_X2(df_md, df_qd, gdp_flash_series, gdp_actual_series, n_lags=4)
    X3, y3     = build_X3(df_md, df_qd, gdp_flash_series, gdp_actual_series)
    X4, y4     = build_X4(df_md, df_qd, gdp_flash_series, gdp_actual_series, n_monthly_lags=4, n_qd_lags=4)
    X_AR, y_AR = build_X_AR(gdp_actual_series, gdp_actual_series, n_lags=2)

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

import pandas as pd
from dateutil.relativedelta import relativedelta

# Real GDP components (consumption, investment, exports) that would let a
# model near-reconstruct GDP if used contemporaneously. Kept for their
# predictive lag information, but never as same-quarter regressors.
GDP_COMPONENT_COLS = ["PCECC96_t_qd", "GPDIC1_t_qd", "EXPGSC1_t_qd"]

def _average_monthly_to_quarterly_from_df(md_filled: pd.DataFrame) -> pd.DataFrame:
    """Averages monthly data to quarterly, indexed by last month of quarter."""
    md = md_filled.copy()
    md["sasdate"] = pd.to_datetime(md["sasdate"])
    md["quarter"] = (
        md["sasdate"]
        .dt.to_period("Q")
        .dt.to_timestamp(how="end")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    return md.drop(columns="sasdate").groupby("quarter").mean()


def _umidas_monthly_to_quarterly_from_df(md_filled: pd.DataFrame) -> pd.DataFrame:
    df = md_filled.copy().sort_values("sasdate").reset_index(drop=True)
    df["sasdate"] = pd.to_datetime(df["sasdate"])
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


def _prep_qd_from_df(qd_filled: pd.DataFrame) -> pd.DataFrame:
    """Sets sasdate as index for quarterly data."""
    qd = qd_filled.copy()
    qd["sasdate"] = pd.to_datetime(qd["sasdate"])
    return qd.set_index("sasdate")


def _build_gdp_lags_from_cut(gdp_cut: pd.Series, index: pd.DatetimeIndex, n_lags: int = 4) -> pd.DataFrame:
    """Builds GDP lag columns aligned to the given index."""
    # reindex first to include target quarter, then shift
    gdp_reindexed = gdp_cut.reindex(index)
    gdp_lags = pd.DataFrame(index=index)
    for lag in range(1, n_lags + 1):
        gdp_lags[f"gdp_lag_{lag}"] = gdp_reindexed.shift(lag)
    return gdp_lags

def _add_lags_df(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """Adds n quarterly lags of every column in df."""
    frames = [df]
    for lag in range(1, n_lags + 1):
        lagged = df.shift(lag).rename(columns={c: f"{c}_lag{lag}" for c in df.columns})
        frames.append(lagged)
    return pd.concat(frames, axis=1)


def _finalise_from_cut(X: pd.DataFrame, gdp_actual: pd.Series) -> tuple:
    """Aligns y to X index using actual GDP."""
    y = gdp_actual.reindex(X.index)
    return X, y

def build_X1_from_cut(
    qd_filled: pd.DataFrame,
    md_filled: pd.DataFrame,
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_lags: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    df_avg     = _average_monthly_to_quarterly_from_df(md_filled)
    df_q       = _prep_qd_from_df(qd_filled).drop(columns=GDP_COMPONENT_COLS, errors="ignore")
    X_base     = df_avg.join(df_q, how="inner")
    gdp_lags   = _build_gdp_lags_from_cut(gdp_cut, X_base.index, n_lags)
    X          = X_base.join(gdp_lags, how="left")
    X, y       = _finalise_from_cut(X, gdp_actual)
    print(f"X1 (avg):            {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


def build_X2_from_cut(
    qd_filled: pd.DataFrame,
    md_filled: pd.DataFrame,
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_lags: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    df_avg     = _average_monthly_to_quarterly_from_df(md_filled)
    df_q       = _prep_qd_from_df(qd_filled)
    X_base     = _add_lags_df(df_avg.join(df_q, how="inner"), n_lags).drop(columns=GDP_COMPONENT_COLS, errors="ignore")
    gdp_lags   = _build_gdp_lags_from_cut(gdp_cut, X_base.index, n_lags)
    X          = X_base.join(gdp_lags, how="left")
    X, y       = _finalise_from_cut(X, gdp_actual)
    print(f"X2 (avg + {n_lags} lags):     {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


def build_X3_from_cut(
    qd_filled: pd.DataFrame,
    md_filled: pd.DataFrame,
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_lags: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    df_umidas  = _umidas_monthly_to_quarterly_from_df(md_filled)

    df_q       = _prep_qd_from_df(qd_filled).drop(columns=GDP_COMPONENT_COLS, errors="ignore")
    X_base     = df_umidas.join(df_q, how="inner")
    gdp_lags   = _build_gdp_lags_from_cut(gdp_cut, X_base.index, n_lags)
    X          = X_base.join(gdp_lags, how="left")
    X, y       = _finalise_from_cut(X, gdp_actual)
    print(f"X3 (U-MIDAS):        {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y

def build_X4_from_cut(
    qd_filled: pd.DataFrame,
    md_filled: pd.DataFrame,
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_monthly_lags: int = 4,
    n_qd_lags: int = 4,
    n_gdp_lags: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    df_umidas        = _umidas_monthly_to_quarterly_from_df(md_filled)
    df_q             = _prep_qd_from_df(qd_filled)

    # add lags to each block independently
    df_umidas_lagged = _add_lags_df(df_umidas, n_monthly_lags)
    df_q_lagged      = _add_lags_df(df_q, n_qd_lags).drop(columns=GDP_COMPONENT_COLS, errors="ignore")

    # use left join so we keep all U‑MIDAS quarters
    X_base = df_umidas_lagged.join(df_q_lagged, how="left")

    # add GDP lags
    gdp_lags = _build_gdp_lags_from_cut(gdp_cut, X_base.index, n_gdp_lags)
    X        = X_base.join(gdp_lags, how="left")

    # finalise, then drop only where *both* X and y are unusable
    X, y = _finalise_from_cut(X, gdp_actual)

    # allow partial‑feature rows, but require y to be observed
    valid = y.notna()
    X = X[valid]
    y = y[valid]

    print(f"X4 (U-MIDAS + lags): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


def build_X_AR_from_cut(
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_lags: int = 2,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Minimal GDP-only AR benchmark with a direct multi-step fallback.

    gdp_cut must be the RAW (un-imputed) GDP series cut at the information
    set available at prediction time — no ensemble/mean fill-in. If the
    quarter immediately before the target (t-1) is released, uses a
    standard 1-step AR(2): lag_a = t-1, lag_b = t-2. If t-1 is NOT yet
    released, uses a direct 2-step forecast instead of imputing the
    missing lag: lag_a = t-2, lag_b = t-3.
    """
    target_date = gdp_cut.index[-1] + relativedelta(months=3)
    full_index  = gdp_cut.index.append(pd.DatetimeIndex([target_date]))
    gdp_reindexed = gdp_cut.reindex(full_index)

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


def build_X_RF_bench_from_cut(
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_lags: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    target_date = gdp_cut.index[-1] + relativedelta(months=3)
    full_index  = gdp_cut.index.append(pd.DatetimeIndex([target_date]))

    gdp_reindexed = gdp_cut.reindex(full_index)
    df = gdp_reindexed.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)
    lag_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    df = df[df[lag_cols].notna().all(axis=1)]
    X  = df[lag_cols]
    y  = gdp_actual.reindex(X.index)
    print(f"X_RF_bench ({n_lags} lags):    {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y

def make_build_X(
    model_name: str,
    n_lags: int = 4,
    n_monthly_lags: int = 4,
    n_qd_lags: int = 4,
):
    """
    Returns a build_X function with the correct signature for the pipeline.
    
    Usage:
        build_fn = make_build_X("X1")
        X, y = build_fn(qd_filled, md_filled, gdp_cut, gdp_actual)
    """
    match model_name:
        case "X1":
            return lambda qd, md, gdp_cut, gdp_actual: build_X1_from_cut(qd, md, gdp_cut, gdp_actual, n_lags)
        case "X2":
            return lambda qd, md, gdp_cut, gdp_actual: build_X2_from_cut(qd, md, gdp_cut, gdp_actual, n_lags)
        case "X3":
            return lambda qd, md, gdp_cut, gdp_actual: build_X3_from_cut(qd, md, gdp_cut, gdp_actual, n_lags)
        case "X4":
            return lambda qd, md, gdp_cut, gdp_actual: build_X4_from_cut(qd, md, gdp_cut, gdp_actual, n_monthly_lags, n_qd_lags)
        case "X_AR":
            return lambda qd, md, gdp_cut, gdp_actual: build_X_AR_from_cut(gdp_cut, gdp_actual)
        case "X_RF_bench":
            return lambda qd, md, gdp_cut, gdp_actual: build_X_RF_bench_from_cut(gdp_cut, gdp_actual, n_lags)
        case _:
            raise ValueError(f"Unknown model_name '{model_name}'. Must be one of: X1, X2, X3, X4, X_AR, X_RF_bench")

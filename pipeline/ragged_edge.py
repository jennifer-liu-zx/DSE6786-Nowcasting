import pandas as pd
from supabase import Client
from statsmodels.tsa.ar_model import AutoReg
from dateutil.relativedelta import relativedelta


def read_table(client: Client, table_name: str) -> pd.DataFrame:
    """Read an entire Supabase table into a DataFrame, handling pagination."""
    all_rows = []
    batch_size = 1000
    offset = 0

    while True:
        response = (
            client.table(table_name)
            .select("*")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        rows = response.data
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < batch_size:
            break
        offset += batch_size

    return pd.DataFrame(all_rows)


def upsert_table(client: Client, table_name: str, df: pd.DataFrame, on_conflict: str = "sasdate"):
    """Upsert a DataFrame into a Supabase table in batches."""
    records = df.where(pd.notnull(df), other=None).to_dict(orient="records")

    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        client.table(table_name).upsert(batch, on_conflict=on_conflict).execute()

    print(f"Upserted {len(records)} rows into '{table_name}'.")


def extend_time_index(df: pd.DataFrame, date_col: str, freq: str, date=None) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if freq == "M":
        freq_pd = "MS"
    elif freq == "Q":
        freq_pd = "QS-MAR"
    else:
        raise ValueError("freq must be 'M' or 'Q'")

    if date is None:
        target_date = pd.Timestamp.today()
    else:
        target_date = pd.Timestamp(date)

    last_day_of_month = target_date + pd.offsets.MonthEnd(0)
    if target_date != last_day_of_month:
        target_date = target_date - pd.offsets.MonthEnd(1)

    target_date = (
        target_date
        .to_period("Q")
        .to_timestamp(how="end")
        .to_period("M")
        .to_timestamp()
    )

    if df[date_col].max() >= target_date:
        return df

    full_range = pd.date_range(start=df[date_col].min(), end=target_date, freq=freq_pd)
    df = df.set_index(date_col).reindex(full_range).reset_index()
    df = df.rename(columns={"index": date_col})

    return df

def _fill_series(series: pd.Series, p: int) -> pd.Series:
    series_filled = series.copy()

    first_obs_idx = series.first_valid_index()
    last_obs_idx = series.last_valid_index()

    if last_obs_idx is None:
        return series

    interpolated = series.loc[first_obs_idx:last_obs_idx].interpolate()
    series_filled.loc[first_obs_idx:last_obs_idx] = interpolated

    n_missing = len(series) - (last_obs_idx + 1)
    if n_missing <= 0:
        return series_filled

    train = interpolated.reset_index(drop=True)
    if len(train) <= p:
        return series_filled

    model = AutoReg(train, lags=p, old_names=False).fit()
    forecast = model.predict(start=len(train), end=len(train) + n_missing - 1)
    series_filled.iloc[last_obs_idx + 1 : last_obs_idx + 1 + n_missing] = forecast.values

    return series_filled


def _assert_no_leftover_nan(df: pd.DataFrame, lag_dict: dict, label: str) -> None:
    """
    _fill_series should resolve every tracked variable (interpolate + AR
    forecast). Any leftover NaN means a variable has no usable history at
    all within this window — silently zero-filling it would corrupt model
    training, so we fail loudly instead.
    """
    tracked_cols = [c for c in lag_dict if c in df.columns]
    nan_counts = df[tracked_cols].isna().sum()
    offending = nan_counts[nan_counts > 0]
    if not offending.empty:
        raise ValueError(
            f"{label}: {len(offending)} variable(s) still have NaN after "
            f"ragged-edge fill (would previously have been silently zeroed): "
            f"{offending.to_dict()}"
        )


def fill_ragged_edge(client: Client, data_table: str, freq: str, date = None) -> pd.DataFrame:
    date_col = "sasdate"
    lag_csv = "data/bic_lags.csv"

    print(f"Reading '{data_table}'...")
    df = read_table(client, data_table)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)
    df = extend_time_index(df, date_col, freq, date)
    print(f"  -> {len(df)} rows after extending to {df[date_col].max().date()}")

    print(f"Reading '{lag_csv}'...")

    lag_df = pd.read_csv(lag_csv)
    lag_dict = dict(zip(lag_df["variable"], lag_df["lag"]))
    print(f"  -> {len(lag_dict)} variables to fill")

    df_filled = df.copy()
    for col in lag_dict:
        if col in df_filled.columns:
            df_filled[col] = pd.to_numeric(df_filled[col], errors="coerce")

    for i, (var, p) in enumerate(lag_dict.items(), 1):
        if var not in df_filled.columns:
            print(f"  [{i}/{len(lag_dict)}] '{var}' skipped (not in DataFrame)")
            continue
        df_filled[var] = _fill_series(df_filled[var], p)

    _assert_no_leftover_nan(df_filled, lag_dict, label=f"fill_ragged_edge({data_table})")
    df_filled = df_filled.fillna(0)
    print(f"Done. Final shape: {df_filled.shape}")

    return df_filled

def fill_ragged_edge_until(QD, MD, cutoff_date):
    QD_target = extend_time_index(QD, "sasdate", "Q", cutoff_date)
    MD_target = extend_time_index(MD, "sasdate", "M", cutoff_date)

    lag_csv = "data/bic_lags.csv"
    lag_df = pd.read_csv(lag_csv)
    lag_dict = dict(zip(lag_df["variable"], lag_df["lag"]))
    print(f"  -> {len(lag_dict)} variables to fill")

    QD_filled = QD_target.copy()
    MD_filled = MD_target.copy()

    for col in lag_dict:
        if col in QD_filled.columns:
            QD_filled[col] = pd.to_numeric(QD_filled[col], errors="coerce")
        elif col in MD_filled.columns:
            MD_filled[col] = pd.to_numeric(MD_filled[col], errors="coerce")

    for i, (var, p) in enumerate(lag_dict.items(), 1):
        if var in QD_filled.columns:
            QD_filled[var] = _fill_series(QD_filled[var], p)
        elif var in MD_filled.columns:
            MD_filled[var] = _fill_series(MD_filled[var], p)
        else:
            print(f"  [{i}/{len(lag_dict)}] '{var}' skipped (not in QD or MD)")

    _assert_no_leftover_nan(QD_filled, lag_dict, label="fill_ragged_edge_until(QD)")
    _assert_no_leftover_nan(MD_filled, lag_dict, label="fill_ragged_edge_until(MD)")
    QD_filled = QD_filled.fillna(0)
    MD_filled = MD_filled.fillna(0)

    return QD_filled, MD_filled

# To test

if __name__ == "__main__": 
    qd = pd.read_csv("data/fred_qd_X.csv")
    md = pd.read_csv("data/fred_md.csv")
    filled_qd, filled_md = fill_ragged_edge_until(qd, md, cutoff_date="2026-06-01")
    print(filled_qd.tail())
    print(filled_md.tail())
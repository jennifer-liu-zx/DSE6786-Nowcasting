import pandas as pd
import numpy as np
from database.client import get_backend_client
from pipeline.evaluation_support import fetch_all_model_forecasts


def get_version(quarter_date: pd.Timestamp, month_date: pd.Timestamp) -> int:
    q_start = quarter_date - pd.DateOffset(months=2)
    offsets = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    diff = (month_date.year - q_start.year) * 12 + (month_date.month - q_start.month)
    if diff not in offsets:
        raise ValueError(
            f"month_date {month_date.date()} is not within 6 months of quarter {quarter_date.date()}"
        )
    return offsets[diff]


def push_forecasts_to_evaluation(client, run_date=None) -> None:
    run_date = str(pd.Timestamp(run_date or pd.Timestamp.today()).date())

    # ── Fetch forecasts ───────────────────────────────────────────────────────
    response = (
        client.table("model_forecasts")
        .select("model_name, quarter_date, month_date, nowcast")
        .eq("run_date", run_date)
        .execute()
    )
    if not response.data:
        print(f"No model_forecasts data found for run_date={run_date}. Skipping.")
        return

    df = fetch_all_model_forecasts(client)

    df["quarter_date"] = pd.to_datetime(df["quarter_date"])
    df["month_date"] = pd.to_datetime(df["month_date"])
    df["nowcast"] = pd.to_numeric(df["nowcast"], errors="coerce")

    # ── EXCLUDE specific quarter_dates ───────────────────────────────────────
    exclude_dates = pd.to_datetime([
        "2020-06-01", "2020-09-01", "2020-12-01"
    ])
    df = df[~df["quarter_date"].isin(exclude_dates)]

    if df.empty:
        print("No data left after excluding specified quarter_dates. Skipping.")
        return

    # ── Safe version computation ──────────────────────────────────────────────
    def safe_get_version(q, m):
        try:
            return get_version(q, m)
        except Exception:
            return None

    df["version"] = df.apply(
        lambda r: safe_get_version(r["quarter_date"], r["month_date"]), axis=1
    )

    df = df.dropna(subset=["version"])
    if df.empty:
        print("No valid rows after version filtering. Skipping.")
        return

    # ── Pivot safely ─────────────────────────────────────────────────────────
    model_cols = [
        "AR_Benchmark", "RF_Lags_Average", "RF_Lags_UMIDAS",
        "LASSO_UMIDAS", "LASSO_Average", "LASSO_Lags_Average",
    ]

    pivot = df.pivot_table(
        index=["quarter_date", "version", "month_date"],
        columns="model_name",
        values="nowcast",
        aggfunc="mean",
        dropna=False
    ).reset_index()

    pivot.columns.name = None

    # Ensure all expected models exist
    for col in model_cols:
        if col not in pivot.columns:
            pivot[col] = pd.NA

    # Compute average skipping missing models
    pivot["All_Model_Average"] = pivot[model_cols].mean(axis=1, skipna=True)

    # Drop rows with all models missing
    pivot = pivot[pivot[model_cols].notna().any(axis=1)]
    if pivot.empty:
        print("No usable rows after filtering empty model data. Skipping.")
        return

    # ── Fetch GDP actuals ────────────────────────────────────────────────────
    gdp_response = client.table("gdp").select("sasdate, GDPC1_t").execute()
    if gdp_response.data:
        gdp_df = pd.DataFrame(gdp_response.data)
        gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
        gdp_series = gdp_df.set_index("sasdate")["GDPC1_t"].astype(float)
        pivot["gdp_actual"] = pivot["quarter_date"].map(gdp_series)
    else:
        pivot["gdp_actual"] = None

    # ── DROP rows with missing GDP ────────────────────────────────────────────
    pivot = pivot.dropna(subset=["gdp_actual"])
    if pivot.empty:
        print("No rows with valid GDP actuals. Skipping upsert.")
        return

    # ── Build records for upsert ─────────────────────────────────────────────
    records = []
    for _, row in pivot.iterrows():
        record = {
            "quarter_date": row["quarter_date"].strftime("%Y-%m-%d"),
            "version": int(row["version"]),
            "month_date": row["month_date"].strftime("%Y-%m-%d"),
            "gdp_actual": float(row["gdp_actual"])  # safe because we dropped NaNs
        }

        for col in model_cols + ["All_Model_Average"]:
            val = row.get(col)
            if pd.notna(val):
                record[col] = float(val)

        records.append(record)

    if not records:
        print("No records to upsert. Skipping.")
        return

    # ── Upsert ───────────────────────────────────────────────────────────────
    client.table("evaluation").upsert(
        records,
        on_conflict="quarter_date,version"
    ).execute()

    print(f"Upserted {len(records)} rows into 'evaluation'.")


def calculate_and_upsert_rmse(client):
    # Fetch all evaluation data
    response = client.table("evaluation").select("*").limit(100000).execute()
    if not response.data:
        print("No data in evaluation table. Skipping RMSE calculation.")
        return

    df = pd.DataFrame(response.data)

    # Inspect column names
    print("Columns in evaluation table:", df.columns.tolist())

    # Ensure required columns exist
    if "version" not in df.columns or "gdp_actual" not in df.columns:
        print("Required columns ('version', 'gdp_actual') not found. Skipping.")
        return

    # ── Convert types ────────────────────────────────────────────────────────
    df["version"] = pd.to_numeric(df["version"], errors="coerce")
    df["gdp_actual"] = pd.to_numeric(df["gdp_actual"], errors="coerce")
    df["quarter_date"] = pd.to_datetime(df["quarter_date"], errors="coerce")

    # ── Filter out unwanted quarter_dates ────────────────────────────────────
    exclude_dates = pd.to_datetime([
        "2020-06-01", "2020-09-01", "2020-12-01"
    ])
    df = df[~df["quarter_date"].isin(exclude_dates)]

    # ── Model columns ────────────────────────────────────────────────────────
    model_cols = [
        "AR_Benchmark", "RF_Lags_Average", "RF_Lags_UMIDAS",
        "LASSO_UMIDAS", "LASSO_Average", "LASSO_Lags_Average", "All_Model_Average"
    ]

    # Keep only model columns that exist
    model_cols = [col for col in model_cols if col in df.columns]

    # ── Melt to long format ──────────────────────────────────────────────────
    df_long = df.melt(
        id_vars=["version", "gdp_actual"],
        value_vars=model_cols,
        var_name="model",
        value_name="forecast"
    )

    # Remove rows with missing forecast or actual
    df_long = df_long.dropna(subset=["forecast", "gdp_actual"])
    if df_long.empty:
        print("No valid forecast/actual pairs. Skipping RMSE calculation.")
        return

    # ── Compute RMSE ─────────────────────────────────────────────────────────
    rmse_df = (
        df_long.groupby(["model", "version"])
        .apply(lambda x: np.sqrt(np.mean((x["forecast"] - x["gdp_actual"])**2)))
        .reset_index(name="rmse")
    )

    # ── Upsert ───────────────────────────────────────────────────────────────
    records = rmse_df.to_dict(orient="records")
    if records:
        client.table("rmse").upsert(records, on_conflict="model,version").execute()
        print(f"Upserted RMSE for {len(records)} model-version combinations.")
    else:
        print("No records to upsert.")

def calculate_mean_rmse_by_model(client):
    # ── Fetch RMSE table ─────────────────────────────────────────────────────
    response = client.table("rmse").select("*").limit(100000).execute()
    if not response.data:
        print("No RMSE data found.")
        return

    df = pd.DataFrame(response.data)
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")

    # ── Group by model and compute mean RMSE ─────────────────────────────────
    mean_rmse = (
        df.groupby("model", as_index=False)["rmse"]
        .mean()
        .rename(columns={"rmse": "rmse"})
    )

    # ── Assign version label (e.g. 0 or 'mean') ──────────────────────────────
    # Since your schema expects NUMERIC, use a special value like 0
    mean_rmse["version"] = 0

    # ── Reorder columns to match table ───────────────────────────────────────
    mean_rmse = mean_rmse[["model", "version", "rmse"]]

    # ── Upsert into rmse table ───────────────────────────────────────────────
    records = mean_rmse.to_dict(orient="records")

    client.table("rmse").upsert(records, on_conflict="model,version").execute()

    print(f"Upserted mean RMSE for {len(records)} models.")

if __name__ == "__main__":
    supabase = get_backend_client()
    push_forecasts_to_evaluation(supabase)
    calculate_and_upsert_rmse(supabase)
    calculate_mean_rmse_by_model(supabase)
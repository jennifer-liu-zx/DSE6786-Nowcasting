import pandas as pd
import numpy as np
from database.client import get_backend_client
from pipeline.evaluation_support import fetch_all_model_forecasts, get_month_date, compute_ci_bounds


def update_ci_columns(client):
    # ── Fetch ALL model_forecasts (no row limit) ─────────────────────────────
    df = fetch_all_model_forecasts(client)
    if df.empty:
        print("No forecasts found.")
        return

    df["quarter_date"] = pd.to_datetime(df["quarter_date"])
    df["month_date"] = pd.to_datetime(df["month_date"])

    # ── Fetch RMSE ───────────────────────────────────────────────────────────
    rmse_data = client.table("rmse").select("*").limit(100000).execute().data
    if not rmse_data:
        print("No RMSE data found.")
        return

    rmse_df = pd.DataFrame(rmse_data)
    rmse_df["version"] = pd.to_numeric(rmse_df["version"], errors="coerce")

    # ── Map version efficiently (vectorized instead of slow loop) ────────────
    def infer_version(row):
        q, m = row["quarter_date"], row["month_date"]
        for v in range(1, 7):
            if get_month_date(q, v) == m:
                return v
        return None

    df["version"] = df.apply(infer_version, axis=1)
    df = df.dropna(subset=["version"])

    # ── Merge with RMSE ──────────────────────────────────────────────────────
    merged = df.merge(
        rmse_df,
        left_on=["model_name", "version"],
        right_on=["model", "version"],
        how="left"
    )

    merged = merged.dropna(subset=["rmse"])
    if merged.empty:
        print("No rows with RMSE match. Skipping.")
        return

    # ── Compute CI ───────────────────────────────────────────────────────────
    merged["ci_50_lb"], merged["ci_50_ub"], merged["ci_80_lb"], merged["ci_80_ub"] = (
        compute_ci_bounds(merged["nowcast"], merged["rmse"])
    )

    # ── SAFE row-by-row update (no overwrite risk) ───────────────────────────
    updates = 0

    for _, row in merged.iterrows():
        client.table("model_forecasts").update({
            "ci_50_lb": float(row["ci_50_lb"]),
            "ci_50_ub": float(row["ci_50_ub"]),
            "ci_80_lb": float(row["ci_80_lb"]),
            "ci_80_ub": float(row["ci_80_ub"]),
        }).eq("id", row["id"]).execute()

        updates += 1

    print(f"Updated CI columns for {updates} rows safely.")


if __name__ == "__main__":
    supabase = get_backend_client()
    update_ci_columns(supabase)
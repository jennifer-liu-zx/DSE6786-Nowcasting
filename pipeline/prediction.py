import pandas as pd
import numpy as np
from database.client import get_backend_client
from pipeline.models.AR_benchmark import ar_model_nowcast
from pipeline.models.rf import randomForest
from pipeline.models.lasso import fit_lasso
from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash
from pipeline.feature_matrix import build_X1, build_X2, build_X3, build_X4, build_X_AR
from pipeline.evaluation_support import fetch_all_model_forecasts, compute_ci_bounds

def assign_version_prev(run_date):
    return (run_date.month - 1) % 3 + 4

def assign_version_latest(run_date):
    return (run_date.month - 1) % 3 + 1

def nowcast_single(model, X: pd.DataFrame, y: pd.Series, gdp: pd.DataFrame, model_name: str, client, run_date = None) -> pd.DataFrame:
    """
    Wraps a nowcast model to produce a single prediction for gdp.index[-2].
    CI is calculated using RMSE fetched from the evaluation table.
    """
    target_idx = gdp.index[-2]
    train_size = 162

    run_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()
    last_day_of_month = (pd.Timestamp(run_date) + pd.offsets.MonthEnd(0)).date()
    if run_date != last_day_of_month:
        run_date = (pd.Timestamp(run_date) - pd.offsets.MonthEnd(1)).date()

    version = assign_version_prev(run_date)

    # Fetch RMSE from evaluation table
    rmse_response = (
        client.table("rmse")
        .select("*")
        .eq("model", model_name)
        .eq("version", version) 
        .execute()
    )

    if not rmse_response.data:
        raise ValueError(f"No RMSE found in evaluation table for model '{model_name}'.")

    rmse = float(rmse_response.data[0]["rmse"])

    # Get integer position of target_idx in X
    target_pos = X.index.get_loc(target_idx)



    # Slice X and y: train_size rows before target + target row itself
    window_start = target_pos - train_size
    X_window = X.iloc[window_start : target_pos + 1]
    y_window = y.iloc[window_start : target_pos + 1]

    # Run model — last row of X_window is the test point
    _, _, _, _, y_test_actual, y_test_predicted = model(X_window, y_window).values()

    lb50, ub50, lb80, ub80 = compute_ci_bounds(float(y_test_predicted), rmse)
    return pd.DataFrame(
        index=[target_idx],
        data={
            "version":       version,
            "quarter":       pd.Period(target_idx, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date(),
            "y_true":        float(y_test_actual),
            "y_hat":         float(y_test_predicted),
            "pred_50_lower": lb50,
            "pred_50_upper": ub50,
            "pred_80_lower": lb80,
            "pred_80_upper": ub80,
        }
    )

def nowcast_single_latest(model, X: pd.DataFrame, y: pd.Series, gdp: pd.DataFrame, model_name: str, client, run_date = None) -> pd.DataFrame:
    """
    Wraps a nowcast model to produce a single prediction for gdp.index[-1].
    If gdp.index[-2] is NA, fetches the latest nowcast for that quarter from Supabase
    and uses it to fill in the missing value before building the training window.
    CI is calculated using RMSE fetched from the evaluation table.
    """
    target_idx = gdp.index[-1]
    prev_idx   = gdp.index[-2]
    train_size = 162

    run_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()
    last_day_of_month = (pd.Timestamp(run_date) + pd.offsets.MonthEnd(0)).date()
    if run_date != last_day_of_month:
        run_date = (pd.Timestamp(run_date) - pd.offsets.MonthEnd(1)).date()

    version = assign_version_latest(run_date)

    # Fetch RMSE from evaluation table
    rmse_response = (
        client.table("rmse")
        .select("*")
        .eq("model", model_name)
        .eq("version", version) 
        .execute()
    )

    if not rmse_response.data:
        raise ValueError(f"No RMSE found in evaluation table for model '{model_name}'.")

    rmse = float(rmse_response.data[0]["rmse"])

    # Fill in gdp.index[-2] if it is NA
    y_filled = y.copy()
    if pd.isna(y.loc[prev_idx]):
        quarter_date = pd.Period(prev_idx, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat()

        response = (
            client.table("model_forecasts")
            .select("nowcast, month_date, run_date")
            .eq("model_name", model_name)
            .eq("quarter_date", quarter_date)
            .order("run_date", desc=True)
            .order("month_date", desc=True)
            .limit(1)
            .execute()  # ← no .single() so 0 rows won't raise
        )

        if response.data:
            fetched_nowcast = float(response.data[0]["nowcast"])
            print(f"  Filling gdp.index[-2] ({prev_idx}) with Supabase nowcast: {fetched_nowcast:.4f} "
                  f"(month: {response.data[0]['month_date']}, run: {response.data[0]['run_date']})")
            y_filled.loc[prev_idx] = fetched_nowcast
        else:
            fallback = float(y.dropna().mean())
            print(f"  No forecast found for {prev_idx} in Supabase — falling back to mean: {fallback:.4f}")
            y_filled.loc[prev_idx] = fallback

    # Get integer position of target_idx in X
    target_pos = X.index.get_loc(target_idx)

    # Slice X and y: train_size rows before target + target row itself
    window_start = target_pos - train_size
    X_window = X.iloc[window_start : target_pos + 1]
    y_window = y_filled.iloc[window_start : target_pos + 1]

    # Run model — last row of X_window is the test point
    _, _, _, _, y_test_actual, y_test_predicted = model(X_window, y_window).values()

    lb50, ub50, lb80, ub80 = compute_ci_bounds(float(y_test_predicted), rmse)
    return pd.DataFrame(
        index=[target_idx],
        data={
            "version":       version,
            "quarter":       pd.Period(target_idx, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date(),
            "y_true":        float(y_test_actual),
            "y_hat":         float(y_test_predicted),
            "pred_50_lower": lb50,
            "pred_50_upper": ub50,
            "pred_80_lower": lb80,
            "pred_80_upper": ub80,
        }
    )

def _push_to_supabase(
    result: pd.DataFrame,
    model_name: str,
    run_date,
    client,
    *,
    push_evaluation: bool,
) -> None:
    run_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()
    """
    Pushes a single model result (one row) to model_forecasts.
    Optionally upserts into evaluation (only for nowcast_single, not latest).
    """
    row = result.iloc[0]
    quarter_date = str(row["quarter"])
    
    month_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()
    last_day_of_month = (pd.Timestamp(month_date) + pd.offsets.MonthEnd(0)).date()
    if month_date != last_day_of_month:
        month_date = (pd.Timestamp(month_date) - pd.offsets.MonthEnd(1)).date()

    month_date = month_date.strftime("%Y-%m-%d")

    # --- model_forecasts ---
    forecast_payload = {
        "run_date":     str(run_date),
        "model_name":   model_name,
        "quarter_date": quarter_date,
        "month_date":   month_date,
        "nowcast":      row["y_hat"],
        "ci_50_lb":     row["pred_50_lower"],
        "ci_50_ub":     row["pred_50_upper"],
        "ci_80_lb":     row["pred_80_lower"],
        "ci_80_ub":     row["pred_80_upper"],
    }
    client.table("model_forecasts").upsert(
    forecast_payload,           # list of dicts or single dict
    on_conflict="model_name,quarter_date,month_date"  # your unique key
    ).execute()


def compute_and_push_model_average(client, quarter_dates: list[str], run_date=None) -> None:
    """
    Computes simple average across all models for the given quarter_dates,
    using only the latest month_date per quarter.
    """
    run_date = str(pd.Timestamp(run_date or pd.Timestamp.today()).date())

    df = fetch_all_model_forecasts(client)

    # Filter for relevant quarters
    df = df[df["quarter_date"].isin(quarter_dates)]

    # Convert numeric columns
    numeric_cols = ["nowcast", "ci_50_lb", "ci_50_ub", "ci_80_lb", "ci_80_ub"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    records = []

    # Group by quarter_date
    for quarter, group in df.groupby("quarter_date"):
        # Select only the latest month_date for this quarter
        latest_month = group["month_date"].max()
        latest_group = group[group["month_date"] == latest_month]

        # Compute average across models
        avg_row = latest_group[numeric_cols].mean()

        records.append({
            "run_date":     run_date,
            "model_name":   "All_Model_Average",
            "quarter_date": quarter,
            "month_date":   latest_month,
            "nowcast":      avg_row["nowcast"],
            "ci_50_lb":     avg_row["ci_50_lb"],
            "ci_50_ub":     avg_row["ci_50_ub"],
            "ci_80_lb":     avg_row["ci_80_lb"],
            "ci_80_ub":     avg_row["ci_80_ub"],
        })

    if records:
        client.table("model_forecasts").upsert(
            records,
            on_conflict="model_name,quarter_date,month_date"
        ).execute()
        print(f"Pushed {len(records)} All_Model_Average records for {quarter_dates}.")
    else:
        print("No records to push.")

def run_all_nowcasts(
    gdp: pd.DataFrame,
    client,
    run_date=None,
) -> None:
    run_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()

    # ── Step 1: historical nowcast (gdp.index[-2]) ──────────────────────────
    print("=== nowcast_single (prev quarter) ===")
    for model_name, cfg in MODEL_REGISTRY.items():
        print(f"  Running {model_name}...")
        result = nowcast_single(
            model=cfg["model"],
            X=cfg["X"], y=cfg["y"], gdp=gdp,
            model_name=model_name,
            client=client,
            run_date=run_date,
        )
        _push_to_supabase(result, model_name, run_date, client, push_evaluation=True)
        print(f"    → {result['y_hat'].iloc[0]:.4f}")

    # ── Step 2: latest nowcast (gdp.index[-1]) ───────────────────────────────
    print("\n=== nowcast_single_latest (current quarter) ===")
    for model_name, cfg in MODEL_REGISTRY.items():
        print(f"  Running {model_name}...")
        result = nowcast_single_latest(
            model=cfg["model"],
            X=cfg["X"], y=cfg["y"], gdp=gdp,
            model_name=model_name,
            client=client,
        )
        _push_to_supabase(result, model_name, run_date, client, push_evaluation=False)
        print(f"    → {result['y_hat'].iloc[0]:.4f}")

def prediction_pipeline(run_date=None):
    df_md, df_qd = load_filled_data()
    gdp_actual_series = load_gdp()["GDPC1_t"]
    gdp_flash_series  = load_gdp_with_flash()
    X_ar, y_ar = build_X_AR(gdp_actual_series, gdp_actual_series, n_lags=2)
    X1, y1 = build_X1(df_md, df_qd, gdp_flash_series, gdp_actual_series)
    X2, y2 = build_X2(df_md, df_qd, gdp_flash_series, gdp_actual_series, n_lags=4)
    X3, y3 = build_X3(df_md, df_qd, gdp_flash_series, gdp_actual_series)
    X4, y4 = build_X4(df_md, df_qd, gdp_flash_series, gdp_actual_series, n_monthly_lags=4, n_qd_lags=4)

    global MODEL_REGISTRY
    MODEL_REGISTRY = {
        "AR_Benchmark": {
            "model": ar_model_nowcast,
            "X": X_ar,
            "y": y_ar,
        },
        "RF_Lags_Average": {
            "model": randomForest,
            "X": X2,
            "y": y2,
        },
        "RF_Lags_UMIDAS": {
            "model": randomForest,
            "X": X4,
            "y": y4,
        },
        "LASSO_UMIDAS": {
            "model": fit_lasso,
            "X": X3,
            "y": y3,
        },
        "LASSO_Average": {
            "model": fit_lasso,
            "X": X1,
            "y": y1,
        },
        "LASSO_Lags_Average": {
            "model": fit_lasso,
            "X": X2,
            "y": y2,
        }
    }
    supabase_client = get_backend_client()
    gdp_response = supabase_client.table("gdp").select("sasdate, GDPC1_t").order("sasdate", desc=False).execute()
    gdp_df = pd.DataFrame(gdp_response.data)
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    gdp_df = gdp_df.set_index("sasdate")


    quarter_dates = [
        pd.Period(gdp_df.index[-2], freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat(),
        pd.Period(gdp_df.index[-1], freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat(),
    ]

    run_all_nowcasts(gdp_df, supabase_client, run_date = run_date)
    compute_and_push_model_average(supabase_client, quarter_dates)

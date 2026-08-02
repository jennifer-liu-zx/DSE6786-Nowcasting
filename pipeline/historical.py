from database.client import get_backend_client
import pandas as pd
import numpy as np
from pipeline.models.AR_benchmark import ar_model_nowcast
from pipeline.models.rf import randomForest
from pipeline.models.lasso import fit_lasso
from pipeline.poos import poos_validation
from pipeline.evaluation_support import get_month_date, plot_poos_results

def push_poos_to_supabase(client, models: list, version: int, run_date=None):
    run_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()

    for model_name, poos_out, rmse, mae in models:
        records = []

        for idx, row in poos_out.iterrows():
            idx = pd.to_datetime(idx)
            quarter_date = (
                idx.to_period("Q").asfreq("M", how="end").to_timestamp()
            )
            month_date = get_month_date(idx, version)

            records.append({
                "run_date":     run_date.strftime("%Y-%m-%d"),
                "model_name":   model_name,
                "quarter_date": quarter_date.strftime("%Y-%m-%d"),
                "month_date":   month_date.strftime("%Y-%m-%d"),
                "nowcast":      None if pd.isna(row["y_hat"])         else float(row["y_hat"]),
                "ci_50_lb":     None if pd.isna(row["pred_50_lower"]) else float(row["pred_50_lower"]),
                "ci_50_ub":     None if pd.isna(row["pred_50_upper"]) else float(row["pred_50_upper"]),
                "ci_80_lb":     None if pd.isna(row["pred_80_lower"]) else float(row["pred_80_lower"]),
                "ci_80_ub":     None if pd.isna(row["pred_80_upper"]) else float(row["pred_80_upper"]),
            })

        if not records:
            print(f"No records to push for model '{model_name}', skipping.")
            continue

        client.table("model_forecasts").upsert(
            records, on_conflict="model_name,quarter_date,month_date"
        ).execute()
        print(f"Upserted {len(records)} POOS record(s) for model '{model_name}' into 'model_forecasts'.")


def push_evaluation_to_supabase(client, models: list, version: int, run_date=None):
    # Pull GDP actuals from Supabase
    gdp_response = client.table("gdp").select("sasdate, GDPC1_t").execute()
    gdp_df = pd.DataFrame(gdp_response.data)
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    gdp_df = gdp_df.set_index("sasdate")

    # Collect y_hat per model
    model_preds = {}
    for model_name, poos_out, rmse, mae in models:
        model_preds[model_name] = poos_out["y_hat"]

    if not model_preds:
        print("No models provided to push_evaluation_to_supabase, skipping.")
        return

    eval_df = pd.DataFrame(model_preds)
    eval_df.index = pd.to_datetime(eval_df.index)

    # Derive month_date from index + version (no longer relies on poos_out["month"])
    eval_df["month_date"] = eval_df.index.map(
        lambda ts: get_month_date(ts, version)
    )

    # Compute average across model columns only
    model_cols = list(model_preds.keys())
    eval_df["All_Model_Average"] = eval_df[model_cols].mean(axis=1)

    # Join GDP actuals — align on quarter-end month timestamp
    gdp_df.index = pd.to_datetime(gdp_df.index)
    eval_df = eval_df.join(gdp_df["GDPC1_t"], how="left")
    eval_df = eval_df.rename(columns={"GDPC1_t": "gdp_actual"})

    # Build records
    records = []
    for idx, row in eval_df.iterrows():
        quarter_date = (
            pd.Timestamp(idx).to_period("Q").asfreq("M", how="end").to_timestamp()
        )
        record = {
            "quarter_date": quarter_date.strftime("%Y-%m-%d"),
            "version":      version,
            "month_date":   pd.Timestamp(row["month_date"]).strftime("%Y-%m-%d"),
            "gdp_actual":   None if pd.isna(row["gdp_actual"]) else float(row["gdp_actual"]),
        }
        for col in model_cols + ["All_Model_Average"]:
            val = row[col]
            record[col] = None if pd.isna(val) else float(val)
        records.append(record)

    client.table("evaluation").upsert(
        records, on_conflict="quarter_date,version"
    ).execute()
    print(f"Upserted {len(records)} rows into 'evaluation' (version={version}).")


BUILD_REGISTRY = {
    "AR_Benchmark":      "X_AR",
    "RF_Lags_Average":   "X2",
    "RF_Lags_UMIDAS":    "X4",
    "LASSO_UMIDAS":      "X3",
    "LASSO_Average":     "X1",
    "LASSO_Lags_Average": "X2",
}

MODEL_REGISTRY: dict[str, dict] = {
    "AR_Benchmark": {
        "model": ar_model_nowcast
    },
    "RF_Lags_Average": {
        "model": randomForest
    },
    "RF_Lags_UMIDAS": {
        "model": randomForest
    },
    "LASSO_UMIDAS": {
        "model": fit_lasso
    },
    "LASSO_Average": {
        "model": fit_lasso

    },
    "LASSO_Lags_Average": {
        "model": fit_lasso
    }
}

def run():
    client = get_backend_client()

    # ── Load data ─────────────────────────────────────────────────────────────
    QD_t = pd.read_csv("data/fred_qd_X.csv")
    QD_t["sasdate"] = pd.to_datetime(QD_t["sasdate"])

    MD_t = pd.read_csv("data/fred_md.csv")
    MD_t["sasdate"] = pd.to_datetime(MD_t["sasdate"])

    gdp_df = pd.read_csv("data/gdp.csv")
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    y_full = gdp_df.set_index("sasdate")["GDPC1_t"]

    # ── Run POOS per version and push ─────────────────────────────────────────
    for version in [4,5,6]:
        print(f"\n{'='*60}\nRunning POOS for version {version}\n{'='*60}")
        poos_results = []

        for model_name, cfg in MODEL_REGISTRY.items():
            build_name = BUILD_REGISTRY[model_name]
            print(f"\n--- {model_name} (buildX={build_name}) ---")

            poos_out, rmse, mae = poos_validation(
                method=cfg["model"],
                buildname=build_name,
                QD_t=QD_t,
                MD_t=MD_t,
                y_full=y_full,
                version=version,
            )

            plot_poos_results(y_full, poos_out, model_name, version)
            print(f"  {model_name} → RMSE={rmse:.4f} | MAE={mae:.4f}")
            poos_results.append((model_name, poos_out, rmse, mae))

        # Push forecasts for all versions
        push_poos_to_supabase(client, poos_results, version=version)

if __name__ == "__main__":
    run()
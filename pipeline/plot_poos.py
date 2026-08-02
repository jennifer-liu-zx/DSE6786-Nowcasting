import pandas as pd
from database.client import get_backend_client
from pipeline.evaluation_support import fetch_all_model_forecasts, get_month_date, plot_poos_results


def fetch_gdp(client) -> pd.Series:
    gdp_response = (
        client.table("gdp")
        .select("sasdate, GDPC1_t")
        .order("sasdate", desc=False)
        .execute()
    )
    gdp_df = pd.DataFrame(gdp_response.data)
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    gdp_df = gdp_df.set_index("sasdate")
    return gdp_df["GDPC1_t"]


def infer_version(quarter_date: pd.Timestamp, month_date: pd.Timestamp) -> int | None:
    # Versions 1-6 represent the six months across and after a quarter
    for v in range(1, 7):
        if get_month_date(quarter_date, v) == month_date:
            return v
    return None


def run():
    client = get_backend_client()

    # Pull GDP actuals
    y_full = fetch_gdp(client)

    # Pull all model forecasts (paginated)
    df = fetch_all_model_forecasts(client)
    if df.empty:
        print("No model forecasts found.")
        return

    df["quarter_date"] = pd.to_datetime(df["quarter_date"])
    df["month_date"] = pd.to_datetime(df["month_date"])

    # Infer version for each row
    df["version"] = df.apply(
        lambda row: infer_version(row["quarter_date"], row["month_date"]), axis=1
    )
    df = df.dropna(subset=["version"])
    df["version"] = df["version"].astype(int)

    # Rename columns to match plot_poos_results expectations
    df = df.rename(
        columns={
            "nowcast": "y_hat",
            "ci_50_lb": "pred_50_lower",
            "ci_50_ub": "pred_50_upper",
            "ci_80_lb": "pred_80_lower",
            "ci_80_ub": "pred_80_upper",
        }
    )

    # Plot for each (model_name, version) pair
    for (model_name, version), group in df.groupby(["model_name", "version"]):
        y_df = group.set_index("quarter_date")[
            ["y_hat", "pred_50_lower", "pred_50_upper", "pred_80_lower", "pred_80_upper"]
        ].sort_index()
        print(f"Plotting {model_name} — version {version} ({len(y_df)} rows)")
        plot_poos_results(y_full, y_df, model_name, version)


if __name__ == "__main__":
    run()
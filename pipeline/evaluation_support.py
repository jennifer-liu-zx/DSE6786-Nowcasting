"""
Shared evaluation-support code used across the prediction and POOS
(pseudo-out-of-sample) pipeline stages: paginated model_forecasts fetch,
version-to-month-date mapping, the Gaussian-quantile confidence-interval
formula, and POOS result plotting.

This module depends on nothing in pipeline.poos, pipeline.historical, or
pipeline.prediction — it sits below all three so any of them can import
from here without creating a circular import.
"""

import pandas as pd


def fetch_all_model_forecasts(client) -> pd.DataFrame:
    """Read the entire model_forecasts table, paginating in batches of 1000."""
    all_rows = []
    page_size = 1000  # Supabase default limit
    start = 0

    while True:
        response = (
            client.table("model_forecasts")
            .select("*")
            .range(start, start + page_size - 1)
            .execute()
        )
        data = response.data or []
        all_rows.extend(data)

        if len(data) < page_size:
            break  # last page
        start += page_size

    return pd.DataFrame(all_rows)


def get_month_date(quarter_ts: pd.Timestamp, version: int) -> pd.Timestamp:
    """
    Given a quarter timestamp and version, return the last day of the target month.

    Version 1 = 1st month of same quarter      (e.g. Q1 -> Jan 31)
    Version 2 = 2nd month of same quarter      (e.g. Q1 -> Feb 28)
    Version 3 = 3rd month of same quarter      (e.g. Q1 -> Mar 31)
    Version 4 = 1st month of next quarter      (e.g. Q1 -> Apr 30)
    Version 5 = 2nd month of next quarter      (e.g. Q1 -> May 31)
    Version 6 = 3rd month of next quarter      (e.g. Q1 -> Jun 30)
    """
    if version not in range(1, 7):
        raise ValueError(f"version must be 1-6, got {version}")

    quarter_start = quarter_ts.to_period("Q").to_timestamp(how="start")

    # Offset in months from quarter start: v1->0, v2->1, v3->2, v4->3, v5->4, v6->5
    month_offset = version - 1
    target = quarter_start + pd.DateOffset(months=month_offset)

    # Return last calendar day of that month
    return target + pd.offsets.MonthEnd(0)


def compute_ci_bounds(point, rmse):
    """
    Gaussian-quantile 50%/80% confidence-interval bounds around a point
    forecast. Works on scalars or pandas Series/arrays (plain arithmetic
    broadcasts either way) -- returns (lb50, ub50, lb80, ub80).
    """
    z50 = 0.674
    z80 = 1.282
    return (
        point - z50 * rmse,
        point + z50 * rmse,
        point - z80 * rmse,
        point + z80 * rmse,
    )


def plot_poos_results(
    y_full: pd.Series,
    y_df: pd.DataFrame,
    model_name: str,
    version: int,
    last_n: int = 200,
) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))

    y_plot = y_full.iloc[-last_n:]
    cutoff_date = y_plot.index[0]

    ax.plot(
        y_plot.index,
        y_plot.values,
        color="black",
        linewidth=1.2,
        label="Actual (full sample)",
        zorder=3,
    )

    y_df_plot = y_df[y_df.index >= cutoff_date]
    idx = y_df_plot.index

    ax.plot(
        idx,
        y_df_plot["y_hat"],
        color="red",
        linewidth=1.2,
        label="Predicted (OOS)",
        zorder=4,
    )

    ax.fill_between(
        idx,
        y_df_plot["pred_50_lower"],
        y_df_plot["pred_50_upper"],
        alpha=0.4,
        color="steelblue",
        label="50% CI",
    )

    ax.fill_between(
        idx,
        y_df_plot["pred_80_lower"],
        y_df_plot["pred_80_upper"],
        alpha=0.2,
        color="steelblue",
        label="80% CI",
    )

    ax.axvline(
        x=idx[0],
        color="grey",
        linestyle=":",
        linewidth=1,
        label="OOS start",
    )

    title = f"{model_name} — Version {version} — POOS Results"

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("GDP growth")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    os.makedirs("pipeline/plots", exist_ok=True)
    safe_title = title.replace(" ", "_").replace("/", "_")
    fig.savefig(
        os.path.join("pipeline/plots", f"{safe_title}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

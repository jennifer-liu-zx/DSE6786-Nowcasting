import os
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

from pipeline.output_x_poos import (
    build_X1_from_cut,
    build_X2_from_cut,
    build_X3_from_cut,
    build_X4_from_cut,
    build_X_AR_from_cut,
    make_build_X
)
from pipeline.ragged_edge import fill_ragged_edge_until
from database.client import get_backend_client

# ── Global constants ─────────────────────────────────────────────────────────

TEST_SIZE  = 100   # OOS quarters (from the last known GDP quarters)
TRAIN_SIZE = 162   # Fixed training window (quarters)


# ── Placeholder benchmark model ──────────────────────────────────────────────

def placeholder_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Unconditional-mean benchmark.

    Treats the last row of X / last element of y as the test observation.
    """
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1].values
    X_test  = X.iloc[[-1]]
    y_test_actual = float(y.iloc[-1])

    y_mean = float(np.mean(y_train))
    y_train_predicted = np.full_like(y_train, y_mean, dtype=float)
    y_test_predicted  = y_mean

    return {
        "X_train":           X_train,
        "y_train":           y_train,
        "y_train_predicted": y_train_predicted,
        "X_test":            X_test,
        "y_test_actual":     y_test_actual,
        "y_test_predicted":  y_test_predicted,
    }


# ── Cut-and-fill snapshot at prediction time ────────────────────────────────

def cut_and_fill(version: int,
                 q_predicted: pd.Timestamp,
                 QD_t: pd.DataFrame,
                 MD_t: pd.DataFrame,
                 gdp: pd.Series,
                 model_name: str = "All_Model_Average",
                 ):

    client = get_backend_client()   

    q_start    = q_predicted - relativedelta(months=2)
    prev_q_end = q_start - relativedelta(months=1)

    version_cutoffs = {
        1: (prev_q_end - relativedelta(months=3),  prev_q_end - relativedelta(months=2), prev_q_end - relativedelta(months=3)),
        2: (prev_q_end - relativedelta(months=3),  prev_q_end - relativedelta(months=1), prev_q_end - relativedelta(months=3)),
        3: (prev_q_end,                            prev_q_end,                            prev_q_end),
        4: (prev_q_end,                            q_start,                               prev_q_end),
        5: (prev_q_end,                            q_start + relativedelta(months=1),     prev_q_end),
        6: (q_predicted,                           q_predicted,                           prev_q_end),
    }

    qd_cutoff, md_cutoff, gdp_cutoff = version_cutoffs[version]

    print(f"Version {version} | Predicting: {q_predicted.date()} | "
          f"QD until: {qd_cutoff.date()} | "
          f"MD until: {md_cutoff.date()} | "
          f"GDP until: {gdp_cutoff.date()}")

    QD_cut = QD_t[QD_t["sasdate"] <= pd.Timestamp(qd_cutoff)].copy()
    MD_cut = MD_t[MD_t["sasdate"] <= pd.Timestamp(md_cutoff)].copy()

    qd_filled, md_filled = fill_ragged_edge_until(QD_cut, MD_cut, cutoff_date=q_predicted)

    # ── Cut GDP at gdp_cutoff, then fill gap up to prev_q_end with nowcasts ──
    gdp_cut = gdp[gdp.index <= pd.Timestamp(gdp_cutoff)].copy()

    quarters_to_fill = pd.date_range(
        start=gdp_cutoff + relativedelta(months=3),
        end=prev_q_end,
        freq="QS-DEC",
    )

    # Raw (un-imputed) cut for the AR benchmark: reindex onto the regular
    # quarterly grid through prev_q_end so that a genuinely un-released
    # quarter (gdp_cutoff < prev_q_end) shows up as a NaN row -- preserving
    # date continuity -- instead of being silently absent from the index.
    # This is what lets build_X_AR_from_cut correctly detect "t-1 not yet
    # released" and fall back to a direct 2-step forecast, without ever
    # touching the ensemble's own predictions. When gdp_cutoff == prev_q_end
    # (the common case), quarters_to_fill is empty and this is a no-op copy.
    gdp_cut_raw = gdp_cut.reindex(gdp_cut.index.append(quarters_to_fill))

    if len(quarters_to_fill) > 0:
        quarter_date_strs = [
            pd.Period(q, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat()
            for q in quarters_to_fill
        ]

        response = (
            client.table("model_forecasts")
            .select("quarter_date, nowcast, run_date")
            .eq("model_name", model_name)
            .in_("quarter_date", quarter_date_strs)
            .order("run_date", desc=True)
            .execute()
        )

        if not response.data:
            print(f"  Warning: no forecasts found to fill GDP gap — leaving as NaN.")
        else:
            fetched = (
                pd.DataFrame(response.data)
                .sort_values("run_date", ascending=False)
                .drop_duplicates(subset="quarter_date")
                .set_index("quarter_date")
            )

            for q, q_date_str in zip(quarters_to_fill, quarter_date_strs):
                if q_date_str in fetched.index:
                    nowcast = float(fetched.loc[q_date_str, "nowcast"])
                    gdp_cut.loc[q] = nowcast
                    print(f"  Filled GDP at {q.date()} with nowcast: {nowcast:.4f}")
                else:
                    print(f"  Warning: no forecast found for {q.date()} — leaving as NaN.")
    
    if len(quarters_to_fill) > 0:
        nan_quarters = [
            q for q in quarters_to_fill
            if q not in gdp_cut.index or pd.isna(gdp_cut.loc[q])
        ]
        if nan_quarters:
            gdp_mean = float(gdp_cut.dropna().mean())
            for q in nan_quarters:
                gdp_cut.loc[q] = gdp_mean
                print(f"  Safeguard: filled GDP at {q.date()} with mean ({gdp_mean:.4f}) — no nowcast available.")
    
    return qd_filled, md_filled, gdp_cut, gdp_cut_raw


# ── POOS validation loop ─────────────────────────────────────────────────────

def poos_validation(
    method: Callable,
    buildname: str,
    QD_t: pd.DataFrame,
    MD_t: pd.DataFrame,
    y_full: pd.Series,
    version: int,
    num_test: int = TEST_SIZE,
    num_train: int = TRAIN_SIZE,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Realistic pseudo–out-of-sample validation.

    For each of the last `num_test` quarters with known GDP:
      1. Cut QD/MD/GDP to information available at that time.
      2. Fill ragged edge up to q_predicted (AR-based).
      3. Build feature matrix from the filled snapshot (buildX).
      4. Train model on the last `num_train` valid quarters before q_predicted.
      5. Predict GDP for q_predicted.
    """
    # Use only released GDP quarters for evaluation
    y_known = y_full[y_full.notna()]
    if len(y_known) < num_test:
        raise ValueError(f"Only {len(y_known)} known GDP quarters, need {num_test}.")
    test_quarters = y_known.index[-num_test:]

    records = []

    for i, q_predicted in enumerate(test_quarters):
        print(f"\n[{i+1}/{num_test}] q_predicted = {q_predicted.date()}")

        # 1. Cut & fill
        qd_filled, md_filled, gdp_filled, gdp_raw = cut_and_fill(
            version=version,
            q_predicted=pd.Timestamp(q_predicted),
            QD_t=QD_t,
            MD_t=MD_t,
            gdp=y_full
        )
        gdp_for_build = gdp_raw if buildname == "X_AR" else gdp_filled

        # 2. Build features from filled snapshot
        buildfn = make_build_X(buildname)

        X, y = buildfn(
            qd=qd_filled,
            md=md_filled,
            gdp_cut=gdp_for_build,
            gdp_actual=y_full,
        )

        # Deduplicate index and truncate to prediction date
        X = X[~X.index.duplicated(keep="last")]
        y = y[~y.index.duplicated(keep="last")]
        X = X[X.index <= q_predicted]
        y = y[y.index <= q_predicted]

        if q_predicted not in X.index:
            print(f"  WARNING: {q_predicted.date()} missing in feature matrix — skipping.")
            continue

        # 3. Fixed-window train/test split (by index)
        X_before = X[X.index < q_predicted]
        y_before = y[y.index < q_predicted]

        # Drop missing y (unreleased GDP) with a 1-D mask
        if isinstance(y_before, pd.DataFrame):
            mask_valid = y_before.iloc[:, 0].notna()
        else:
            mask_valid = y_before.notna()

        X_before = X_before[mask_valid]
        y_before = y_before[mask_valid]

        if len(X_before) == 0:
            print(f"  WARNING: no valid y_before for {q_predicted.date()} — skipping.")
            continue

        # Fixed *max* training window: shrink if not enough history
        window_size = min(num_train, len(X_before))
        X_train = X_before.iloc[-window_size:]
        y_train = y_before.iloc[-window_size:]

        if len(X_train) == 0:
            print(f"  WARNING: empty training set for {q_predicted.date()} — skipping.")
            continue

        # 4. Build window with last row as test
        X_window = pd.concat([X_train, X.loc[[q_predicted]]])
        y_window = pd.concat([y_train, y.loc[[q_predicted]]])

        X_window = X_window[~X_window.index.duplicated(keep="last")]
        y_window = y_window[~y_window.index.duplicated(keep="last")]

        # 4. Optional: check for NaNs before model fit and skip if present

        # X_window: DataFrame → use .any().any()
        has_nan_X = bool(X_window.isna().any().any())

        # y_window: could be Series or 1-col DataFrame, normalise to Series
        if isinstance(y_window, pd.DataFrame):
            has_nan_y = bool(y_window.isna().any().any())
        else:  # Series
            has_nan_y = bool(y_window.isna().any())

        if has_nan_X or has_nan_y:
            print(f"  WARNING: NaNs detected before model fit for {q_predicted.date()}")

            if has_nan_X:
                nan_cols = X_window.columns[X_window.isna().any()].tolist()
                print("    Columns with NaNs in X_window:")
                print(nan_cols)
                print("    NaN counts by column in X_window:")
                print(X_window.isna().sum()[X_window.isna().sum() > 0])

            if has_nan_y:
                if isinstance(y_window, pd.DataFrame):
                    print("    y_window NaN entries:")
                    print(y_window[y_window.isna().any(axis=1)])
                else:
                    print("    y_window NaN entries:")
                    print(y_window[y_window.isna()])

            print("    Skipping this quarter.")
            continue

        # 5. Fit model and compute training RMSE
        result = method(X_window, y_window)
        (
            _,
            y_train_actual,
            y_train_predicted,
            _,
            y_test_actual,
            y_test_predicted,
        ) = result.values()

        y_train_actual = np.asarray(y_train_actual, dtype=float)
        y_train_predicted = np.asarray(y_train_predicted, dtype=float)
        train_rmse = float(np.sqrt(np.mean((y_train_actual - y_train_predicted) ** 2)))

        # Store OOS result
        y_test_actual = float(y_test_actual)
        y_test_predicted = float(y_test_predicted)

        records.append(
            {
                "index":          q_predicted,
                "y_true":         y_test_actual,
                "y_hat":          y_test_predicted,
                "pred_50_lower":  y_test_predicted - 0.674 * train_rmse,
                "pred_50_upper":  y_test_predicted + 0.674 * train_rmse,
                "pred_80_lower":  y_test_predicted - 1.282 * train_rmse,
                "pred_80_upper":  y_test_predicted + 1.282 * train_rmse,
            }
        )

    if not records:
        raise RuntimeError("No test quarters were successfully evaluated.")

    y_df = pd.DataFrame(records).set_index("index").sort_index()
    rmse = float(np.sqrt(np.mean((y_df["y_true"] - y_df["y_hat"]) ** 2)))
    mae  = float(np.mean(np.abs(y_df["y_true"] - y_df["y_hat"])))

    print(f"\nPOOS complete — {len(y_df)} quarters | RMSE={rmse:.4f} | MAE={mae:.4f}")

    return y_df, rmse, mae


# ── Plotting helper ──────────────────────────────────────────────────────────

def plot_poos_results(
    y_full: pd.Series,
    y_df: pd.DataFrame,
    title: str = "POOS Forecast vs Actual",
    last_n: int = 200,
) -> None:
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

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    qd = pd.read_csv("data/fred_qd_X.csv")
    md = pd.read_csv("data/fred_md.csv")

    qd["sasdate"] = pd.to_datetime(qd["sasdate"], errors="coerce")
    md["sasdate"] = pd.to_datetime(md["sasdate"], errors="coerce")
    # gdp = get_backend_client().table("gdp").select("sasdate, GDPC1_t").execute()
    # gdp_df = pd.DataFrame(gdp.data)
    gdp_df = pd.read_csv("data/gdp.csv")
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"], errors="coerce")
    gdp_df = gdp_df.set_index("sasdate")

    # Smoke-test cut_and_fill
    filled_qd, filled_md, gdp_filled, gdp_raw = cut_and_fill(
        version=4,
        q_predicted=pd.Timestamp("2025-12-01"),
        QD_t=qd,
        MD_t=md,
        gdp=gdp_df["GDPC1_t"]
    )
    print("QD tail:"); print(filled_qd.tail())
    print("MD tail:"); print(filled_md.tail())
    print("GDP tail (filled):"); print(gdp_filled.tail())
    print("GDP tail (raw, used by AR benchmark):"); print(gdp_raw.tail())

    buildX = make_build_X("X1")
    X, y = buildX(filled_qd, filled_md, gdp_filled, gdp_df["GDPC1_t"])

    print("Feature matrix tail:"); print(X.tail())
    print("Target series tail:"); print(y.tail())
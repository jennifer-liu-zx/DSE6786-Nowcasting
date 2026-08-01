import numpy as np
from scipy import stats
import pandas as pd
import numpy as np
from itertools import combinations
from dotenv import load_dotenv
from database.client import get_backend_client
from pipeline.load_data import save_df
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
# import matplotlib.pyplot as plt

load_dotenv()

def compare_model_pairs(df, time_col='quarter_date', **dm_kwargs):
    """
    Runs pairwise DM tests ONLY for models with matching versions.
    Returns a winner-first table with a dedicated version column.
    """
    df = df.copy()
    
    # Create the unique ID (e.g., 'AR_Benchmark_v1')
    df['id'] = df['model_name'] + "_v" + df['version'].astype(str)

    # 1. Pivot to align all models by time
    predictions = df.pivot_table(
        index=time_col,
        columns='id',
        values='nowcast',
        aggfunc='first'
    )

    # 2. Extract aligned actuals
    actuals = (
        df.drop_duplicates(subset=[time_col])
        .set_index(time_col)['gdp_actual']
        .reindex(predictions.index)
    )

    unique_ids = sorted(predictions.columns)
    results = []

    # 3. Iterate through unique pairs
    for id_a, id_b in combinations(unique_ids, 2):
        
        # Split by '_v' and grab the last element (the number)
        ver_a = int(id_a.split('_v')[-1])
        ver_b = int(id_b.split('_v')[-1])

        # FILTER: Only compare if versions match
        if ver_a != ver_b:
            continue

        yhat_a = predictions[id_a]
        yhat_b = predictions[id_b]

        # Use only overlapping time periods
        mask = yhat_a.notna() & yhat_b.notna()
        if mask.sum() < 10:
            continue

        try:
            stat, pval = dm_test(
                actuals[mask].values,
                yhat_a[mask].values,
                yhat_b[mask].values,
                **dm_kwargs
            )

            # --- WINNER-FIRST LOGIC ---
            if stat <= 0:
                m1, m2 = id_a, id_b
                final_stat = stat
            else:
                m1, m2 = id_b, id_a
                final_stat = -stat

            results.append({
                'version': ver_a,
                'model_1': m1,
                'model_2': m2,
                'test_statistic': final_stat,
                'p_value': pval
            })
        
        except Exception:
            continue
    
    results = pd.DataFrame(results)
    results["model_1"] = results["model_1"].str[:-3]
    results["model_2"] = results["model_2"].str[:-3]

    # Return sorted by version, then by highest significance (lowest p-value)
    return results.sort_values(['version', 'p_value'])

def dm_test(
    y_actual: np.ndarray,
    y_hat1: np.ndarray,
    y_hat2: np.ndarray,
    loss: str = "absolute",
    h: int = 1,
    power: float = 2.0,
    bandwidth: str = "auto",
) -> tuple[float, float]:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    Tests H0: E[d_t] = 0, where d_t = L(e1_t) - L(e2_t) is the loss
    differential between forecast 1 and forecast 2.

    A negative test statistic favours forecast 1 (lower loss);
    a positive statistic favours forecast 2.

    Parameters
    ----------
    y_actual : array-like, shape (n,)
        Realised values.
    y_hat1 : array-like, shape (n,)
        Forecasts from model 1.
    y_hat2 : array-like, shape (n,)
        Forecasts from model 2.
    loss : {"squared", "absolute", "power"}
        Loss function applied to each forecast error.
        - "squared"  : L(e) = e²          (default, MSE-based)
        - "absolute" : L(e) = |e|          (MAE-based)
        - "power"    : L(e) = |e|^power    (generalised, set `power` as needed)
    h : int, default 1
        Forecast horizon. Used when bandwidth="fixed" to set HAC lags = h-1.
    power : float, default 2.0
        Exponent used when loss="power". Ignored for other loss choices.
    bandwidth : {"auto", "fixed"}
        HAC lag selection rule.
        - "auto"  : uses floor(P^(1/3)) — the data-driven rule (matches the slide).
        - "fixed" : uses h-1 — the theoretically motivated choice for an
                    h-step forecast error (MA(h-1) structure).

    Returns
    -------
    dm_stat : float
        The DM test statistic ~ N(0,1) asymptotically.
    p_value : float
        Two-sided p-value. Uses t(n-1) with Harvey-Leybourne-Newbold (1997)
        small-sample correction when n < 50, otherwise standard Normal.

    Raises
    ------
    ValueError
        If arrays differ in length, contain NaNs, or arguments are invalid.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> y   = rng.standard_normal(200)
    >>> h1  = y + rng.standard_normal(200) * 0.5   # better forecast
    >>> h2  = y + rng.standard_normal(200) * 1.0
    >>> dm_stat, p_val = diebold_mariano(y, h1, h2)
    >>> print(f"DM = {dm_stat:.4f}, p = {p_val:.4f}")
    """
    y_actual = np.asarray(y_actual, dtype=float)
    y_hat1   = np.asarray(y_hat1,   dtype=float)
    y_hat2   = np.asarray(y_hat2,   dtype=float)

    # ── Validation ────────────────────────────────────────────────────────────
    if not (y_actual.shape == y_hat1.shape == y_hat2.shape):
        raise ValueError("y_actual, y_hat1, and y_hat2 must have the same shape.")
    if np.any(np.isnan(y_actual) | np.isnan(y_hat1) | np.isnan(y_hat2)):
        raise ValueError("Inputs contain NaN values. Remove or impute before testing.")
    if h < 1:
        raise ValueError("Forecast horizon h must be >= 1.")
    if bandwidth not in {"auto", "fixed"}:
        raise ValueError(f"bandwidth must be 'auto' or 'fixed'; got '{bandwidth}'.")

    n = len(y_actual)

    # ── Loss differentials ────────────────────────────────────────────────────
    e1 = y_actual - y_hat1
    e2 = y_actual - y_hat2

    loss_fn = {
        "squared":  lambda e: e ** 2,
        "absolute": lambda e: np.abs(e),
        "power":    lambda e: np.abs(e) ** power,
    }
    if loss not in loss_fn:
        raise ValueError(f"loss must be one of {list(loss_fn)}; got '{loss}'.")

    d     = loss_fn[loss](e1) - loss_fn[loss](e2)
    d_bar = d.mean()
    
    result = adfuller(d)
    print(f'ADF Statistic: {result[0]}')
    print(f'ADF p-value: {result[1]}')

    # ── HAC lag length ────────────────────────────────────────────────────────
    if bandwidth == "auto":
        # P^(1/3) rule — data-driven, consistent with the slide
        n_lags = int(np.floor(n ** (1 / 3)))
    else:
        # Fixed lags for test sample of ~100
        n_lags = 4

    # Regression on constant with HAC standard error following Newey-West implentation in lecture

    X = np.ones(n) 
    results = sm.OLS(d, X).fit(cov_type='HAC', cov_kwds={'maxlags': n_lags, 'use_correction': True})
    dm_stat = results.tvalues[0]

    # ── Harvey-Leybourne-Newbold (1998) small-sample correction ──────────────
    print(n) # sanity check for n
    dm_stat_corrected = (1+(n**-1)*(1-2*h)+(n**-2)*h*(h-1))**0.5 * dm_stat
    p_value  = 2.0 * stats.t.sf(np.abs(dm_stat_corrected), df=n - 1)

    return float(dm_stat_corrected), float(p_value)


def fetch_forecast_data(table_name: str = "evaluation") -> pd.DataFrame:
    """
    Pulls all forecast data from Supabase and returns a cleaned DataFrame.
    """
    supabase = get_backend_client()
    response = supabase.table(table_name).select("quarter_date",
                                                 "version", 
                                                 "gdp_actual","AR_Benchmark","RF_Lags_Average",
                                                 "RF_Lags_UMIDAS", "LASSO_UMIDAS",
                                                 "LASSO_Average", "LASSO_Lags_Average",
                                                 "All_Model_Average").execute()
    
    # Convert to DataFrame
    df = pd.DataFrame(response.data)
    df_long = df.melt(
        id_vars=['quarter_date', 'version', 'gdp_actual'], 
        value_vars=["AR_Benchmark","RF_Lags_Average","RF_Lags_UMIDAS", "LASSO_UMIDAS",
                    "LASSO_Average", "LASSO_Lags_Average","All_Model_Average"],
        var_name='model_name', 
        value_name='nowcast'
)
    
    return df_long

def push_dm_results_to_supabase(client, dm_results_df: pd.DataFrame, table_name="dm_test"):
    """
    Pushes the 5-column DM results DataFrame into a Supabase table.
    """
    records = []

    for _, row in dm_results_df.iterrows():
        records.append({
            "version":        int(row["version"]),
            "model_1":        str(row["model_1"]),
            "model_2":        str(row["model_2"]),
            # Convert Pandas NaN to Python None for JSON serialization
            "test_statistic": None if pd.isna(row["test_statistic"]) else float(row["test_statistic"]),
            "p_value":        None if pd.isna(row["p_value"]) else float(row["p_value"]),
        })

    if not records:
        print("No DM comparison records to push, skipping.")
        return

    try:
        # Conflict relies entirely on the unique pairing of models and versions
        conflict_cols = "version,model_1,model_2"
        
        client.table(table_name).upsert(
            records, 
            on_conflict=conflict_cols
        ).execute()
        
        print(f"Upserted {len(records)} DM comparison record(s) into '{table_name}'.")
        
    except Exception as e:
        print(f"Failed to push DM results to Supabase: {e}")


def main():
    # Initialize Supabase Client
    supabase = get_backend_client()

    # Fetch Forecast Data
    print("Fetching data from Supabase...")
    df_forecasts = fetch_forecast_data()
    
    if df_forecasts.empty:
        print("No data fetched. Exiting pipeline.")
        return

    # Run Pairwise DM Tests
    print("Running Diebold-Mariano tests...")
    # Using 'quarter_date' to align the forecasts 
    dm_results_df = compare_model_pairs(
        df=df_forecasts, 
        time_col='quarter_date', 
    )

    if dm_results_df.empty:
        print("No valid comparisons generated. Check data overlap/alignment.")
        return

    print(dm_results_df.head())

    # Push to Supabase
    print("Pushing results to Supabase...")
    push_dm_results_to_supabase(
        client=supabase, 
        dm_results_df=dm_results_df, 
        table_name="dm_test"
    )

    save_df(dm_results_df, "../data", "dm_test_results")
    print(f"Pipeline complete! Local copy saved .")

if __name__ == "__main__":
    main()

# ── TEST ──────────────────────────────────────────────────────────────────────
# df_forecasts = fetch_forecast_data()
# print(df_forecasts.head(15))
# model_pairs = compare_model_pairs(
#     df_forecasts,
#     time_col='quarter_date'
# )

# from pipeline.load_data import save_df
# save_df(model_pairs, "../data", "filter_dm_test_results")




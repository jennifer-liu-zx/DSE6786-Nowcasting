"""
GDP-loading helpers used to assemble the two point-in-time GDP series that
pipeline.feature_matrix's builders take as parameters: a (possibly
flash-filled) series for constructing gdp_lag* columns, and the raw series
used only as the prediction target (y).

Only the live nowcasting path (pipeline.prediction, pipeline.correlation_check)
uses this module. The POOS path (pipeline.poos) constructs its own
point-in-time GDP cuts from historical vintages instead of fetching the
latest data from Supabase.
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

THIS_DIR    = Path(__file__).resolve().parent   # pipeline/
PROJECT_DIR = THIS_DIR.parent                   # project root

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_DIR))
from ragged_edge import read_table
from database.client import get_backend_client


def load_filled_data():
    """Fetch filled_md and filled_qd from Supabase."""
    print("Loading filled data from Supabase …")
    supabase = get_backend_client()

    df_md = read_table(supabase, "filled_md")
    df_md["sasdate"] = pd.to_datetime(df_md["sasdate"])
    df_md = df_md.sort_values("sasdate").reset_index(drop=True)
    print(f"  filled_md : {df_md.shape}")

    df_qd = read_table(supabase, "filled_qd")
    df_qd["sasdate"] = pd.to_datetime(df_qd["sasdate"])
    df_qd = df_qd.sort_values("sasdate").reset_index(drop=True)
    print(f"  filled_qd : {df_qd.shape}\n")

    return df_md, df_qd


def load_gdp() -> pd.DataFrame:
    """Fetch the raw (un-imputed) GDP table from Supabase, indexed by sasdate."""
    supabase = get_backend_client()
    gdp = read_table(supabase, "gdp")
    gdp["sasdate"] = pd.to_datetime(gdp["sasdate"])
    gdp = gdp.set_index("sasdate").sort_index()
    gdp = gdp[gdp.index.notna()]
    return gdp


def load_gdp_with_flash() -> pd.Series:
    """
    Returns the GDP growth series (GDPC1_t) with any unreleased quarters
    filled by the latest Ensemble flash prediction from model_forecasts.

    Edge case: on 31 March 2026, Q4 2025 GDP may not yet be officially
    released. We substitute the Ensemble nowcast so lag features can be
    constructed for the Q1 2026 nowcast row.
    """
    gdp = load_gdp()
    y = gdp["GDPC1_t"].copy()

    missing = y[y.isna()].index
    if len(missing) == 0:
        return y

    supabase = get_backend_client()
    for date in missing:
        resp = (supabase.table("model_forecasts")
                .select("nowcast")
                .eq("model_name", "All_Model_Average")
                .eq("quarter_date", date.strftime("%Y-%m-%d"))
                .order("month_date", desc=True)
                .execute())
        if resp.data:
            flash_val = float(resp.data[0]["nowcast"])
            y[date] = flash_val
            print(f"  GDP lag: using Ensemble flash prediction "
                  f"{flash_val:.4f} for {date.date()} (not yet officially released)")
        else:
            print(f"  GDP lag: no flash prediction found for {date.date()}, "
                  f"lag will remain NaN")
    return y

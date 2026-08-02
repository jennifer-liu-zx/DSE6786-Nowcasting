# Feature-Matrix Builder Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the duplicated live (`pipeline/output_x.py`) and POOS (`pipeline/output_x_poos.py`) feature-matrix builders into two new modules — a pure transformation module and a Supabase-fetching module — fixing a real join-semantics bug in `build_X4` along the way, then delete both old files.

**Architecture:** `pipeline/gdp_data.py` (new) holds the Supabase-fetching helpers that assemble the two point-in-time GDP series every builder needs. `pipeline/feature_matrix.py` (new) holds the actual builders (`build_X1`–`build_X4`, `build_X_AR`, `build_X_AR_from_cut`) as pure functions with zero Supabase awareness — every GDP series is passed in as a parameter. Four call sites (`prediction.py`, `poos.py`, `historical.py`, `correlation_check.py`) are updated to match, and `output_x.py`/`output_x_poos.py` are deleted.

**Tech Stack:** Python 3.13, pandas, python-dateutil.

## Global Constraints

- `build_X1`, `build_X2`, `build_X3`, `build_X4` take `(df_md, df_qd, gdp_for_lags, gdp_actual, ...)` — `gdp_for_lags` builds the `gdp_lag*` feature columns, `gdp_actual` is the raw series used only for `y`. No wrapper object.
- `build_X4` switches its internal join from `how="left"` (POOS's old behavior) to `how="inner"` (matching live's existing behavior) — this is an intentional behavior change, not a bug carried forward. It recovers 18 POOS evaluations (2021-09 through 2025-12) for the production `RF_Lags_UMIDAS` model that are currently lost because a single missing raw FRED-MD value (`CP3Mx`/`COMPAPFFx` at 2020-06) poisons every downstream training window that still contains it. `build_X4` also gains an explicit `n_gdp_lags: int = 4` parameter (previously hardcoded in the live version).
- `build_X_AR` (live-style: assumes the input series' own last row already is the target/nowcast row) and `build_X_AR_from_cut` (POOS-style: extends the input series' index by one quarter to synthesize the target row) stay as two distinct functions, not unified — they encode genuinely different assumptions about their caller's input shape, and `build_X_AR`'s fallback logic was the site of a real, recently-fixed production bug (multi-quarter-unreleased-GDP handling). Do not attempt to merge their internals. Both still take `(gdp_for_lags, gdp_actual, n_lags=2)`.
- `build_X_RF_bench_from_cut` is deleted outright — confirmed dead code, no caller reaches it via `make_build_X`.
- Canonical positional argument order is `(df_md, df_qd, ...)` — monthly before quarterly — everywhere, replacing POOS's old `(qd, md, ...)` order.
- `make_build_X` moves to `pipeline/poos.py` (its only real consumer) rather than into `feature_matrix.py` — it's a POOS-orchestration dispatch concern, not a feature-matrix concern.
- No pytest/unittest exists in this repo — verification is standalone `python3` scripts with `assert` statements, matching the existing `if __name__ == "__main__":` idiom.
- Work happens in an isolated git worktree off current `main` (`5226b58`), at `.claude/worktrees/feature-matrix-unification`. Every implementer must verify `pwd`/`git rev-parse HEAD`/`git branch --show-current` before doing anything, and again before committing.
- Any verification script using `sys.path.insert(...)` must point at the worktree checkout path above, not the main repo path.
- Supabase is paused — no live DB access. All behavioral verification uses the local fixtures at `/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data/` (`fred_md.csv`, `fred_qd_X.csv`, `gdp.csv` — these are the correct, non-stale column-naming stand-ins for `filled_md`/`filled_qd`, per this session's earlier finding). Verification touching Supabase-fetching code (`pipeline/gdp_data.py`) is import/signature-only.
- A fresh Python venv is needed (this session's scratchpad venv won't persist): `pandas`, `numpy`, `python-dateutil`, `python-dotenv`, `supabase`, `statsmodels` (the last is a transitive need — `pipeline/ragged_edge.py`, imported by `pipeline/gdp_data.py`, imports `statsmodels.tsa.ar_model`).

---

## Task 1: Create `pipeline/gdp_data.py`

**Files:**
- Create: `pipeline/gdp_data.py`
- Test: standalone verification script (scratchpad)

**Interfaces:**
- Produces: `load_filled_data() -> tuple[pd.DataFrame, pd.DataFrame]` (returns `df_md, df_qd`), `load_gdp() -> pd.DataFrame` (raw `gdp` table, indexed by `sasdate`), `load_gdp_with_flash() -> pd.Series` (GDP growth with unreleased quarters filled from the Ensemble flash prediction).

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification")

import inspect
from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash

assert list(inspect.signature(load_filled_data).parameters) == []
assert list(inspect.signature(load_gdp).parameters) == []
assert list(inspect.signature(load_gdp_with_flash).parameters) == []

# None of these should touch Supabase at import time (only when called) —
# confirm the module imports cleanly with no network activity.
import pipeline.gdp_data
print("ALL GDP_DATA CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `ModuleNotFoundError: No module named 'pipeline.gdp_data'`

- [ ] **Step 3: Create `pipeline/gdp_data.py`**

```python
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
```

- [ ] **Step 4: Run the verification script again to confirm it passes**

Expected: `ALL GDP_DATA CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/gdp_data.py
git commit -m "Add pipeline/gdp_data.py: Supabase GDP/filled-data loaders"
```

---

## Task 2: Create `pipeline/feature_matrix.py`

**Files:**
- Create: `pipeline/feature_matrix.py`
- Test: standalone verification script (scratchpad), using local data fixtures

**Interfaces:**
- Consumes: nothing from Task 1 (pure module — GDP series are passed in by callers).
- Produces: `build_X1(df_md, df_qd, gdp_for_lags, gdp_actual) -> tuple[pd.DataFrame, pd.Series]`, `build_X2(df_md, df_qd, gdp_for_lags, gdp_actual, n_lags=4) -> tuple`, `build_X3(df_md, df_qd, gdp_for_lags, gdp_actual) -> tuple`, `build_X4(df_md, df_qd, gdp_for_lags, gdp_actual, n_monthly_lags=4, n_qd_lags=4, n_gdp_lags=4) -> tuple`, `build_X_AR(gdp_for_lags, gdp_actual, n_lags=2) -> tuple`, `build_X_AR_from_cut(gdp_for_lags, gdp_actual, n_lags=2) -> tuple`.

- [ ] **Step 1: Write the failing verification script**

This reuses the local data fixtures already confirmed to work in this session (`/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data/`). It checks three things: (a) the module imports and has the right signatures, (b) `build_X1`/`build_X2`/`build_X3` produce shapes consistent with the old live builders (unaffected by this refactor), and (c) `build_X4`'s join-semantics fix is real — it recovers the 18 quarters currently lost to the CP3Mx/COMPAPFFx gap.

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification")

import inspect
import pandas as pd
from pipeline.feature_matrix import build_X1, build_X2, build_X3, build_X4, build_X_AR, build_X_AR_from_cut

# ── Signature checks ─────────────────────────────────────────────────────────
assert list(inspect.signature(build_X1).parameters) == ["df_md", "df_qd", "gdp_for_lags", "gdp_actual"]
assert list(inspect.signature(build_X4).parameters) == [
    "df_md", "df_qd", "gdp_for_lags", "gdp_actual", "n_monthly_lags", "n_qd_lags", "n_gdp_lags"
]
assert list(inspect.signature(build_X_AR).parameters) == ["gdp_for_lags", "gdp_actual", "n_lags"]
assert list(inspect.signature(build_X_AR_from_cut).parameters) == ["gdp_for_lags", "gdp_actual", "n_lags"]

# ── Load local fixtures ──────────────────────────────────────────────────────
# Note: df_md's sasdate must be pre-converted to datetime here — production
# code relies on load_filled_data() having already done this; _prep_qd
# self-converts for df_qd, but the monthly-data helpers do not.
DATA_DIR = "/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data"
md = pd.read_csv(f"{DATA_DIR}/fred_md.csv")
md["sasdate"] = pd.to_datetime(md["sasdate"])
qd = pd.read_csv(f"{DATA_DIR}/fred_qd_X.csv")
gdp_df = pd.read_csv(f"{DATA_DIR}/gdp.csv")
gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
gdp = gdp_df.set_index("sasdate").sort_index()["GDPC1_t"]

# No flash data available locally (Supabase paused) — use raw gdp for both
# roles, which is a valid (if slightly degenerate) point-in-time pair.
X1, y1 = build_X1(md, qd, gdp, gdp)
X2, y2 = build_X2(md, qd, gdp, gdp, n_lags=4)
X3, y3 = build_X3(md, qd, gdp, gdp)
assert X1.shape[0] > 200 and X1.notna().all(axis=1).all()
assert X2.shape[0] > 200 and X2.notna().all(axis=1).all()
assert X3.shape[0] > 200 and X3.notna().all(axis=1).all()

# ── X4 join-semantics fix ────────────────────────────────────────────────────
# build_X4's own inner-join + _finalise already drop every row with any
# missing feature, so X4/y4 are guaranteed NaN-free by construction — that's
# the actual fix (a left join would instead let a handful of incomplete rows
# survive into X, which is what let one missing raw value poison ~18 unrelated
# POOS evaluations downstream in poos.py's windowing logic; see the grilling
# session notes in this plan's Global Constraints for the full trace).
X4, y4 = build_X4(md, qd, gdp, gdp, n_monthly_lags=4, n_qd_lags=4, n_gdp_lags=4)
assert X4.notna().all(axis=1).all(), "build_X4 must drop all NaN rows (inner-join behavior)"
assert X4.shape[0] > 200, f"expected most of the ~266-quarter history to survive, got {X4.shape[0]} rows"

# Confirm the known-bad quarters (single missing raw CP3Mx/COMPAPFFx value at
# 2020-06, propagated through 4 lag shifts) are excluded from X4's index
# rather than surviving with NaN cells.
known_bad_quarters = pd.to_datetime(["2020-06-01", "2020-09-01", "2020-12-01", "2021-03-01", "2021-06-01"])
still_present = [d for d in known_bad_quarters if d in X4.index]
assert still_present == [], f"expected the known-incomplete quarters to be dropped, but found: {still_present}"

print(f"X4 shape after inner-join fix: {X4.shape}")
print("ALL FEATURE_MATRIX CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `ModuleNotFoundError: No module named 'pipeline.feature_matrix'`

- [ ] **Step 3: Create `pipeline/feature_matrix.py`**

```python
"""
feature_matrix.py
==================
Prepares feature matrices (X, y) for all model specifications, for both the
live nowcasting path and POOS (pseudo-out-of-sample) evaluation.

Every builder is a pure function: no Supabase access, no filesystem access.
Callers (pipeline.prediction, pipeline.poos) are responsible for assembling
the two GDP series each builder needs — see pipeline.gdp_data for the
live-path loaders, and pipeline.poos.cut_and_fill for the POOS-path cuts.

  gdp_for_lags — feeds the gdp_lag* feature columns. Flash-filled for
                 X1-X4 (an unreleased quarter's lag can use the ensemble's
                 own flash prediction); raw/point-in-time-cut for the AR
                 benchmarks, which must never depend on the ensemble.
  gdp_actual   — the raw series used only to look up y (never flash-filled).

Four datasets
-------------
  X1 — Simple average
        Monthly variables averaged within each quarter, joined with quarterly data.
        Shape: (n_quarters, n_monthly + n_quarterly)

  X2 — Simple average + lags
        Same as X1 (call it qd1), then add 4 quarterly lags of every column in qd1.
        Shape: (n_quarters, (n_monthly + n_quarterly) × 5)

  X3 — U-MIDAS
        Monthly variables kept as 3 separate features per quarter (_m1/_m2/_m3),
        joined with quarterly data.
        Shape: (n_quarters, n_monthly×3 + n_quarterly)

  X4 — U-MIDAS + lags
        U-MIDAS monthly block (current quarter) + 4 quarterly lags of that block
        (= 12 monthly observations per variable), plus quarterly data + 4 lags.
        Shape: (n_quarters, n_monthly×3×5 + n_quarterly×5)
"""

import pandas as pd
from dateutil.relativedelta import relativedelta

# Real GDP components (consumption, investment, exports) that would let a
# model near-reconstruct GDP if used contemporaneously. Kept for their
# predictive lag information, but never as same-quarter regressors.
GDP_COMPONENT_COLS = ["PCECC96_t_qd", "GPDIC1_t_qd", "EXPGSC1_t_qd"]


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _average_monthly_to_quarterly(df_md: pd.DataFrame) -> pd.DataFrame:
    """
    Average the 3 monthly observations within each quarter per variable.
    Excludes COVID flag columns (kept as quarterly-level features in df_qd).
    Returns DataFrame indexed by quarter label, columns suffixed with _md.
    """
    df = df_md.copy().sort_values("sasdate").reset_index(drop=True)
    feature_cols = [c for c in df.columns
                    if c not in ("sasdate", "covid_crash", "covid_recover")]
    df["qtr_label"] = (
        df["sasdate"].dt.to_period("Q").dt.start_time
        + pd.DateOffset(months=2)
    )
    df_agg = (
        df.groupby("qtr_label")[feature_cols]
        .mean()
        .rename_axis("sasdate")
        .add_suffix("_md")
    )
    return df_agg


def _umidas_monthly_to_quarterly(df_md: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly data to quarterly U-MIDAS features.
    Each variable becomes 3 columns: _m1 (oldest), _m2, _m3 (most recent).
    Excludes COVID flag columns.
    Returns DataFrame indexed by quarter label.
    """
    df = df_md.copy().sort_values("sasdate").reset_index(drop=True)
    df["qtr_label"] = (
        df["sasdate"].dt.to_period("Q").dt.start_time
        + pd.DateOffset(months=2)
    )
    df["month_pos"] = df.groupby("qtr_label").cumcount() + 1
    feature_cols = [c for c in df.columns
                    if c not in ("sasdate", "qtr_label", "month_pos",
                                 "covid_crash", "covid_recover")]
    df_pivot = df.pivot(index="qtr_label", columns="month_pos", values=feature_cols)
    df_pivot.columns = [f"{col}_m{pos}" for col, pos in df_pivot.columns]
    df_pivot.index.name = "sasdate"
    return df_pivot


def _prep_qd(df_qd: pd.DataFrame) -> pd.DataFrame:
    df = df_qd.copy()
    df["sasdate"] = pd.to_datetime(df["sasdate"])
    return df.set_index("sasdate").sort_index()


def _add_lags(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """
    Add n_lags quarterly lags of every column in df.
    Lag-k columns are named {col}_lag{k}.
    The lag-0 (current) columns keep their original names.
    """
    lagged = [df]
    for k in range(1, n_lags + 1):
        shifted = df.shift(k).add_suffix(f"_lag{k}")
        lagged.append(shifted)
    return pd.concat(lagged, axis=1)


def _build_gdp_lags(gdp_for_lags: pd.Series, index: pd.DatetimeIndex, n_lags: int = 4) -> pd.DataFrame:
    """Builds GDP lag columns (gdp_lag1..gdp_lag{n_lags}) aligned to index."""
    reindexed = gdp_for_lags.reindex(index)
    out = pd.DataFrame(index=index)
    for k in range(1, n_lags + 1):
        out[f"gdp_lag{k}"] = reindexed.shift(k)
    return out


def _finalise(X: pd.DataFrame, gdp_actual: pd.Series) -> tuple:
    """
    Align y to X's index, drop any row where X has an incomplete feature.

    Rows where X is complete but y is NaN are retained only if they fall
    after the last known GDP date — these are the nowcast rows (e.g. 2026 Q1).
    Rows before the GDP series begins are dropped.

    NOTE: y will be NaN for nowcast rows. When passing to poos_validation(),
    use only rows where y.notna() for evaluation, and use the last row of X
    separately for the actual nowcast prediction.
    """
    X = X.reindex(gdp_actual.index)
    y = gdp_actual
    valid = X.notna().all(axis=1)
    if (~valid).sum() > 0:
        print(f"  Dropping {(~valid).sum()} rows with NaNs.")
    return X[valid], y[valid]


# =============================================================================
# X1 — SIMPLE AVERAGE
# =============================================================================

def build_X1(df_md: pd.DataFrame, df_qd: pd.DataFrame,
             gdp_for_lags: pd.Series, gdp_actual: pd.Series) -> tuple:
    """
    X1: averaged monthly features + quarterly features.

    Monthly variables are averaged across the 3 months within each quarter.
    """
    df_avg = _average_monthly_to_quarterly(df_md)
    df_q   = _prep_qd(df_qd)
    missing = [c for c in GDP_COMPONENT_COLS if c not in df_q.columns]
    if missing:
        raise ValueError(f"GDP component columns not found in quarterly data: {missing}")
    df_q = df_q.drop(columns=GDP_COMPONENT_COLS)
    X = df_avg.join(df_q, how="inner")
    X = X.join(_build_gdp_lags(gdp_for_lags, X.index), how="left")
    X, y = _finalise(X, gdp_actual)
    print(f"X1 (avg):            {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X2 — SIMPLE AVERAGE + LAGS
# =============================================================================

def build_X2(df_md: pd.DataFrame, df_qd: pd.DataFrame,
             gdp_for_lags: pd.Series, gdp_actual: pd.Series, n_lags: int = 4) -> tuple:
    """
    X2: averaged monthly + quarterly (call it qd1), then add n_lags quarterly
    lags of every column in qd1.

    Total features = (n_monthly_avg + n_quarterly) × (1 + n_lags)
    """
    df_avg = _average_monthly_to_quarterly(df_md)
    df_q   = _prep_qd(df_qd)
    qd1 = df_avg.join(df_q, how="inner")
    X = _add_lags(qd1, n_lags).drop(columns=GDP_COMPONENT_COLS, errors="ignore")
    X = X.join(_build_gdp_lags(gdp_for_lags, X.index), how="left")
    X, y = _finalise(X, gdp_actual)
    print(f"X2 (avg + {n_lags} lags):     {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X3 — U-MIDAS
# =============================================================================

def build_X3(df_md: pd.DataFrame, df_qd: pd.DataFrame,
             gdp_for_lags: pd.Series, gdp_actual: pd.Series) -> tuple:
    """
    X3: U-MIDAS monthly features (_m1/_m2/_m3) + quarterly features.

    Each monthly variable becomes 3 quarterly features preserving
    within-quarter dynamics.
    """
    df_umidas = _umidas_monthly_to_quarterly(df_md)
    df_q      = _prep_qd(df_qd)
    missing = [c for c in GDP_COMPONENT_COLS if c not in df_q.columns]
    if missing:
        raise ValueError(f"GDP component columns not found in quarterly data: {missing}")
    df_q = df_q.drop(columns=GDP_COMPONENT_COLS)
    X = df_umidas.join(df_q, how="inner")
    X = X.join(_build_gdp_lags(gdp_for_lags, X.index), how="left")
    X, y = _finalise(X, gdp_actual)
    print(f"X3 (U-MIDAS):        {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X4 — U-MIDAS + LAGS
# =============================================================================

def build_X4(df_md: pd.DataFrame, df_qd: pd.DataFrame,
             gdp_for_lags: pd.Series, gdp_actual: pd.Series,
             n_monthly_lags: int = 4, n_qd_lags: int = 4, n_gdp_lags: int = 4) -> tuple:
    """
    X4: U-MIDAS monthly block + n_monthly_lags quarterly lags of that block,
    plus quarterly data + n_qd_lags quarterly lags.

    n_monthly_lags=4 means 4 quarterly shifts of the _m1/_m2/_m3 block,
    covering 4×3=12 monthly observations of history per variable.

    n_qd_lags=4 means 4 quarterly lags of each quarterly variable.

    Uses an INNER join between the U-MIDAS monthly block and the quarterly
    block (matching the live nowcast path's historical behavior) — a row
    is only kept if every feature is present. A prior POOS-only version of
    this function used a left join intended to "keep all U-MIDAS quarters,"
    but that interacted badly with downstream POOS windowing logic that
    discards an entire evaluation quarter if any NaN survives anywhere in
    its training window: one missing raw value could silently poison up to
    TRAIN_SIZE quarters' worth of unrelated evaluations. Inner join instead
    drops exactly the row(s) that are actually incomplete.
    """
    df_umidas = _umidas_monthly_to_quarterly(df_md)
    df_q      = _prep_qd(df_qd)

    df_umidas_lagged = _add_lags(df_umidas, n_monthly_lags)
    df_q_lagged      = _add_lags(df_q, n_qd_lags).drop(columns=GDP_COMPONENT_COLS, errors="ignore")

    X = df_umidas_lagged.join(df_q_lagged, how="inner")
    X = X.join(_build_gdp_lags(gdp_for_lags, X.index, n_gdp_lags), how="left")
    X, y = _finalise(X, gdp_actual)
    print(f"X4 (U-MIDAS + lags): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X_AR — AUTOREGRESSIVE BENCHMARK (2 GDP lags)
# =============================================================================

def build_X_AR(gdp_for_lags: pd.Series, gdp_actual: pd.Series, n_lags: int = 2) -> tuple:
    """
    X_AR: minimal GDP-only AR benchmark, direct-forecast on real lags only.

    Live-path version: assumes gdp_for_lags' own last row already IS the
    target/nowcast quarter (its value may be NaN if not yet released) —
    this matches how the Supabase gdp table is maintained (a placeholder
    row exists for the current quarter before official release). For the
    POOS-path equivalent, which must synthesize that target row itself
    from a deliberately-truncated historical cut, see build_X_AR_from_cut.

    Never uses a flash-filled series — the benchmark must not depend on
    the ensemble's own predictions. If GDP at t-1 is released, uses a
    standard 1-step AR(2) (lag_a=t-1, lag_b=t-2). If t-1 is not yet
    released, uses a direct 2-step forecast (lag_a=t-2, lag_b=t-3) instead
    of imputing the missing lag.
    """
    df = gdp_for_lags.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 2):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)

    t1_available = pd.notna(df["lag_1"].iloc[-1])
    lag_cols = ["lag_1", "lag_2"] if t1_available else ["lag_2", "lag_3"]

    df_hist = df.iloc[:-1][["gdp_growth"] + lag_cols]
    df_hist = df_hist[df_hist[lag_cols].notna().all(axis=1)]
    X_hist = df_hist[lag_cols].rename(columns={lag_cols[0]: "lag_a", lag_cols[1]: "lag_b"})
    y_hist = df_hist["gdp_growth"]

    last_row = df.iloc[[-1]]
    X_last = last_row[lag_cols].rename(columns={lag_cols[0]: "lag_a", lag_cols[1]: "lag_b"})
    y_last = last_row["gdp_growth"]

    X = pd.concat([X_hist, X_last])
    y = pd.concat([y_hist, y_last])
    print(f"X_AR (direct-forecast, using {lag_cols}): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


def build_X_AR_from_cut(gdp_for_lags: pd.Series, gdp_actual: pd.Series, n_lags: int = 2) -> tuple:
    """
    X_AR: minimal GDP-only AR benchmark, direct-forecast on real lags only.

    POOS-path version: gdp_for_lags must be the RAW (un-imputed) GDP series
    cut at the information set available at prediction time — no
    ensemble/mean fill-in — and, unlike build_X_AR, does NOT already
    contain a row for the target quarter (it's a historical vintage that
    stops one quarter short). This function synthesizes that target row
    by extending the index by one quarter before applying the same
    t-1-availability fallback as build_X_AR. See build_X_AR's docstring
    for why these two are not merged into one function.
    """
    target_date = gdp_for_lags.index[-1] + relativedelta(months=3)
    full_index  = gdp_for_lags.index.append(pd.DatetimeIndex([target_date]))
    gdp_reindexed = gdp_for_lags.reindex(full_index)

    df = gdp_reindexed.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 2):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)

    t1_available = pd.notna(df["lag_1"].iloc[-1])
    lag_cols = ["lag_1", "lag_2"] if t1_available else ["lag_2", "lag_3"]

    df = df[df[lag_cols].notna().all(axis=1)]
    X = df[lag_cols].rename(columns={lag_cols[0]: "lag_a", lag_cols[1]: "lag_b"})
    y = gdp_actual.reindex(X.index)
    print(f"X_AR (direct-forecast, using {lag_cols}): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# MAIN — live-path CSV export, mirrors the old output_x.py script
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_DIR / "data"

    df_md, df_qd = load_filled_data()
    gdp_actual_series = load_gdp()["GDPC1_t"]
    gdp_flash_series  = load_gdp_with_flash()

    X1, y1     = build_X1(df_md, df_qd, gdp_flash_series, gdp_actual_series)
    X2, y2     = build_X2(df_md, df_qd, gdp_flash_series, gdp_actual_series, n_lags=4)
    X3, y3     = build_X3(df_md, df_qd, gdp_flash_series, gdp_actual_series)
    X4, y4     = build_X4(df_md, df_qd, gdp_flash_series, gdp_actual_series, n_monthly_lags=4, n_qd_lags=4)
    X_AR, y_AR = build_X_AR(gdp_actual_series, gdp_actual_series, n_lags=2)

    datasets = [
        ("X1",       X1,       y1),
        ("X2",       X2,       y2),
        ("X3",       X3,       y3),
        ("X4",       X4,       y4),
        ("X_AR",     X_AR,     y_AR),
    ]

    print("\n=== Saving to CSV ===")
    for name, X, y in datasets:
        x_path = DATA_DIR / f"{name}.csv"
        y_path = DATA_DIR / f"y_{name}.csv"
        X.to_csv(x_path)
        y.to_csv(y_path, header=True)
        print(f"  {name}: X={X.shape} → {x_path.name}, y={y.shape} → {y_path.name}")
```

- [ ] **Step 4: Run the verification script again to confirm it passes**

Expected: `ALL FEATURE_MATRIX CHECKS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add pipeline/feature_matrix.py
git commit -m "Add pipeline/feature_matrix.py: unified builders, fix build_X4 join semantics"
```

---

## Task 3: Migrate `pipeline/prediction.py`

**Files:**
- Modify: `pipeline/prediction.py:7` (imports), `pipeline/prediction.py:280-286` (`prediction_pipeline`)

**Interfaces:**
- Consumes: `load_filled_data`, `load_gdp`, `load_gdp_with_flash` from `pipeline.gdp_data` (Task 1); `build_X1`, `build_X2`, `build_X3`, `build_X4`, `build_X_AR` from `pipeline.feature_matrix` (Task 2).

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification")

import ast
src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification/pipeline/prediction.py").read()
ast.parse(src)
assert "pipeline.output_x" not in src, "prediction.py still imports from the old output_x module"
assert "from pipeline.gdp_data import" in src
assert "from pipeline.feature_matrix import" in src
assert "build_X_AR()" not in src, "build_X_AR must be called with explicit gdp arguments now"

print("ALL TASK 3 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: prediction.py still imports from the old output_x module`

- [ ] **Step 3: Edit `pipeline/prediction.py`**

Replace line 7:
```python
from pipeline.output_x import build_X1, build_X2, build_X3, build_X4, load_filled_data, build_X_AR, load_filled_data
```
with:
```python
from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash
from pipeline.feature_matrix import build_X1, build_X2, build_X3, build_X4, build_X_AR
```

Replace `prediction_pipeline`'s opening lines (currently 280-286):
```python
def prediction_pipeline(run_date=None):
    df_md, df_qd = load_filled_data()
    X_ar , y_ar = build_X_AR()
    X1, y1 = build_X1(df_md, df_qd)
    X2, y2 = build_X2(df_md, df_qd, n_lags=4)
    X3, y3 = build_X3(df_md, df_qd)
    X4, y4 = build_X4(df_md, df_qd, n_monthly_lags=4, n_qd_lags=4)
```
with:
```python
def prediction_pipeline(run_date=None):
    df_md, df_qd = load_filled_data()
    gdp_actual_series = load_gdp()["GDPC1_t"]
    gdp_flash_series  = load_gdp_with_flash()
    X_ar, y_ar = build_X_AR(gdp_actual_series, gdp_actual_series, n_lags=2)
    X1, y1 = build_X1(df_md, df_qd, gdp_flash_series, gdp_actual_series)
    X2, y2 = build_X2(df_md, df_qd, gdp_flash_series, gdp_actual_series, n_lags=4)
    X3, y3 = build_X3(df_md, df_qd, gdp_flash_series, gdp_actual_series)
    X4, y4 = build_X4(df_md, df_qd, gdp_flash_series, gdp_actual_series, n_monthly_lags=4, n_qd_lags=4)
```

Nothing else in `prediction.py` changes — the rest of the file references `X_ar`/`y_ar`/`X1`/`y1`/etc. by name, which still resolve.

- [ ] **Step 4: Run the verification script again**

Expected: `ALL TASK 3 CHECKS PASSED`

- [ ] **Step 5: Confirm the file still parses and imports cleanly**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification")
import ast
ast.parse(open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification/pipeline/prediction.py").read())
print("prediction.py OK")
```

- [ ] **Step 6: Commit**

```bash
git add pipeline/prediction.py
git commit -m "Migrate prediction.py onto pipeline.feature_matrix and pipeline.gdp_data"
```

---

## Task 4: Migrate `pipeline/poos.py` and `pipeline/historical.py`

**Files:**
- Modify: `pipeline/poos.py:10-19` (imports), `pipeline/poos.py:199-207` (`buildfn` call in `poos_validation`), `pipeline/poos.py:330-359` (`__main__`)
- Modify: `pipeline/historical.py:7` (delete dead import)

**Interfaces:**
- Consumes: `build_X1`, `build_X2`, `build_X3`, `build_X4`, `build_X_AR_from_cut` from `pipeline.feature_matrix` (Task 2).
- Produces: `make_build_X(model_name, n_lags=4, n_monthly_lags=4, n_qd_lags=4)` now lives in `pipeline.poos` (moved from the deleted `output_x_poos.py`).

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification")

import ast
src_poos = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification/pipeline/poos.py").read()
ast.parse(src_poos)
assert "pipeline.output_x_poos" not in src_poos, "poos.py still imports from the old output_x_poos module"
assert "def make_build_X(" in src_poos, "make_build_X should now be defined in poos.py"
assert "gdp_for_lags=" in src_poos, "poos_validation should call buildfn with the new gdp_for_lags keyword"

src_hist = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification/pipeline/historical.py").read()
ast.parse(src_hist)
assert "pipeline.output_x_poos" not in src_hist, "historical.py still imports from the old output_x_poos module"

print("ALL TASK 4 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: poos.py still imports from the old output_x_poos module`

- [ ] **Step 3: Edit `pipeline/poos.py` imports**

Replace lines 10-17:
```python
from pipeline.output_x_poos import (
    build_X1_from_cut,
    build_X2_from_cut,
    build_X3_from_cut,
    build_X4_from_cut,
    build_X_AR_from_cut,
    make_build_X
)
```
with:
```python
from pipeline.feature_matrix import build_X1, build_X2, build_X3, build_X4, build_X_AR_from_cut
```

- [ ] **Step 4: Add `make_build_X` to `pipeline/poos.py`**

Insert this function directly above `poos_validation` (i.e., right after `placeholder_model` and before the `# ── Cut-and-fill snapshot at prediction time ──` comment, or anywhere else at module scope before its first use — placement relative to `cut_and_fill` doesn't matter since Python resolves names at call time):

```python
def make_build_X(
    model_name: str,
    n_lags: int = 4,
    n_monthly_lags: int = 4,
    n_qd_lags: int = 4,
):
    """
    Returns a build_X function with the uniform (md, qd, gdp_for_lags,
    gdp_actual) signature poos_validation calls regardless of buildname.

    Usage:
        build_fn = make_build_X("X1")
        X, y = build_fn(md, qd, gdp_for_lags, gdp_actual)
    """
    match model_name:
        case "X1":
            return lambda md, qd, gdp_for_lags, gdp_actual: build_X1(md, qd, gdp_for_lags, gdp_actual)
        case "X2":
            return lambda md, qd, gdp_for_lags, gdp_actual: build_X2(md, qd, gdp_for_lags, gdp_actual, n_lags)
        case "X3":
            return lambda md, qd, gdp_for_lags, gdp_actual: build_X3(md, qd, gdp_for_lags, gdp_actual)
        case "X4":
            return lambda md, qd, gdp_for_lags, gdp_actual: build_X4(md, qd, gdp_for_lags, gdp_actual, n_monthly_lags, n_qd_lags)
        case "X_AR":
            return lambda md, qd, gdp_for_lags, gdp_actual: build_X_AR_from_cut(gdp_for_lags, gdp_actual)
        case _:
            raise ValueError(f"Unknown model_name '{model_name}'. Must be one of: X1, X2, X3, X4, X_AR")
```

(This is the same dispatcher that used to live in `output_x_poos.py`, minus the `X_RF_bench` case — that builder is deleted as dead code — and with `n_lags` never passed to the `X_AR` case, matching the existing fix that keeps the AR benchmark at a fixed 2 lags regardless of the outer `n_lags` used for X1-X4.)

- [ ] **Step 5: Update the `buildfn` call in `poos_validation`**

Replace (currently lines 199-207):
```python
        # 2. Build features from filled snapshot
        buildfn = make_build_X(buildname)

        X, y = buildfn(
            qd=qd_filled,
            md=md_filled,
            gdp_cut=gdp_for_build,
            gdp_actual=y_full,
        )
```
with:
```python
        # 2. Build features from filled snapshot
        buildfn = make_build_X(buildname)

        X, y = buildfn(
            md=md_filled,
            qd=qd_filled,
            gdp_for_lags=gdp_for_build,
            gdp_actual=y_full,
        )
```

- [ ] **Step 6: Fix the positional call in `poos.py`'s `__main__`**

The old lambda signature was `(qd, md, gdp_cut, gdp_actual)`; the new one is `(md, qd, gdp_for_lags, gdp_actual)` — the smoke test's positional call must swap its first two arguments or it will silently pass `filled_qd` where `md` is expected. Replace (currently lines 355-356):
```python
    buildX = make_build_X("X1")
    X, y = buildX(filled_qd, filled_md, gdp_filled, gdp_df["GDPC1_t"])
```
with:
```python
    buildX = make_build_X("X1")
    X, y = buildX(filled_md, filled_qd, gdp_filled, gdp_df["GDPC1_t"])
```

- [ ] **Step 7: Delete the dead import in `pipeline/historical.py`**

Delete line 7 entirely:
```python
from pipeline.output_x_poos import make_build_X, build_X1_from_cut, build_X2_from_cut, build_X3_from_cut, build_X4_from_cut, build_X_AR_from_cut, build_X_RF_bench_from_cut
```
Nothing in `historical.py` actually calls any of these names directly (it only calls `poos_validation`, which now owns its own `make_build_X` internally) — this import has been dead code since before this refactor. No replacement import is needed.

- [ ] **Step 8: Run the verification script again**

Expected: `ALL TASK 4 CHECKS PASSED`

- [ ] **Step 9: Behavioral smoke test against local fixtures**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification")
import subprocess
result = subprocess.run(
    ["python3", "pipeline/poos.py"],
    cwd="/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification",
    capture_output=True, text=True, timeout=120,
)
print(result.stdout[-2000:])
print(result.stderr[-2000:])
assert "Feature matrix tail:" in result.stdout, "poos.py's __main__ smoke test did not complete"
assert result.returncode == 0
print("POOS SMOKE TEST OK")
```

- [ ] **Step 10: Commit**

```bash
git add pipeline/poos.py pipeline/historical.py
git commit -m "Migrate poos.py and historical.py onto pipeline.feature_matrix; move make_build_X into poos.py"
```

---

## Task 5: Migrate `pipeline/correlation_check.py`

**Files:**
- Modify: `pipeline/correlation_check.py:30-34` (imports), `pipeline/correlation_check.py:47-54` (data loading/building), `pipeline/correlation_check.py:74` (comment)

**Interfaces:**
- Consumes: `load_filled_data`, `load_gdp`, `load_gdp_with_flash` from `pipeline.gdp_data` (Task 1); `build_X1`, `build_X2`, `build_X3`, `build_X4`, `build_X_AR` from `pipeline.feature_matrix` (Task 2).

This script was already broken before this refactor — `build_X_RF_bench` never existed in `output_x.py`, so importing it always raised `ImportError`. The `build_X_RF_bench()` call's result (`X_rf_bench`, `y_rf_bench`) was never used anywhere else in the file either (confirmed via `grep` during grilling — it appears once, at assignment, and nowhere else). This task removes the dead RF-benchmark scaffolding and repoints the rest of the imports so the script becomes importable again; its actual PCA/correlation analysis logic (all of it keyed off `X1`/`y1`) is untouched.

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification")

import ast
src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification/pipeline/correlation_check.py").read()
ast.parse(src)
assert "pipeline.output_x" not in src, "correlation_check.py still imports from the old output_x module"
assert "build_X_RF_bench" not in src, "dead RF-bench references not fully removed"

print("ALL TASK 5 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: correlation_check.py still imports from the old output_x module`

- [ ] **Step 3: Edit `pipeline/correlation_check.py`**

Replace the import block (currently lines 30-34):
```python
from pipeline.output_x import (
    load_filled_data,
    build_X1, build_X2, build_X3, build_X4,
    build_X_AR, build_X_RF_bench,
)
```
with:
```python
from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash
from pipeline.feature_matrix import (
    build_X1, build_X2, build_X3, build_X4,
    build_X_AR,
)
```

Replace the data-loading/building block (currently lines 47-54):
```python
df_md, df_qd = load_filled_data()

X_ar,       y_ar       = build_X_AR()
X_rf_bench, y_rf_bench = build_X_RF_bench()
X1,         y1         = build_X1(df_md, df_qd)
X2,         y2         = build_X2(df_md, df_qd)
X3,         y3         = build_X3(df_md, df_qd)
X4,         y4         = build_X4(df_md, df_qd)
```
with:
```python
df_md, df_qd = load_filled_data()
gdp_actual_series = load_gdp()["GDPC1_t"]
gdp_flash_series  = load_gdp_with_flash()

X_ar, y_ar = build_X_AR(gdp_actual_series, gdp_actual_series, n_lags=2)
X1,   y1   = build_X1(df_md, df_qd, gdp_flash_series, gdp_actual_series)
X2,   y2   = build_X2(df_md, df_qd, gdp_flash_series, gdp_actual_series)
X3,   y3   = build_X3(df_md, df_qd, gdp_flash_series, gdp_actual_series)
X4,   y4   = build_X4(df_md, df_qd, gdp_flash_series, gdp_actual_series)
```

Replace the comment on line 74:
```python
# X_ar and X_rf_bench only use GDP lags — no macro variables to drop
```
with:
```python
# X_ar only uses GDP lags — no macro variables to drop
```

- [ ] **Step 4: Run the verification script again**

Expected: `ALL TASK 5 CHECKS PASSED`

- [ ] **Step 5: Confirm the file parses**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification")
import ast
ast.parse(open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification/pipeline/correlation_check.py").read())
print("correlation_check.py OK")
```

- [ ] **Step 6: Commit**

```bash
git add pipeline/correlation_check.py
git commit -m "Migrate correlation_check.py onto pipeline.feature_matrix; remove dead RF-benchmark scaffolding"
```

---

## Task 6: Delete `pipeline/output_x.py` and `pipeline/output_x_poos.py`; full-repo sweep

**Files:**
- Delete: `pipeline/output_x.py`, `pipeline/output_x_poos.py`

**Interfaces:**
- Consumes: nothing new — this task only removes the now-fully-superseded old modules and confirms no stragglers reference them.

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification")

import os
worktree = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification"
assert os.path.exists(f"{worktree}/pipeline/output_x.py"), "expected output_x.py to still exist before this task runs"
assert os.path.exists(f"{worktree}/pipeline/output_x_poos.py"), "expected output_x_poos.py to still exist before this task runs"
print("PRE-DELETE STATE CONFIRMED")
```

- [ ] **Step 2: Run it to confirm it passes (sanity check, not a real failing-test step — there's nothing to "fail" before a deletion)**

Expected: `PRE-DELETE STATE CONFIRMED`

- [ ] **Step 3: Delete the old modules**

```bash
git rm pipeline/output_x.py pipeline/output_x_poos.py
```

- [ ] **Step 4: Full-repo consistency sweep**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification")
import subprocess

worktree = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification"

# No file anywhere should still reference the deleted modules.
out = subprocess.run(
    ["grep", "-rl", "output_x", "pipeline/", "app.py"],
    cwd=worktree, capture_output=True, text=True,
).stdout.strip().splitlines()
assert out == [], f"stale references to output_x/output_x_poos found: {out}"

# Every touched (and untouched-but-adjacent) module still parses and imports.
import ast
files = [
    "pipeline/gdp_data.py", "pipeline/feature_matrix.py",
    "pipeline/prediction.py", "pipeline/poos.py", "pipeline/historical.py",
    "pipeline/correlation_check.py",
]
for f in files:
    src = open(f"{worktree}/{f}").read()
    ast.parse(src)
    print(f, "parses OK")

result = subprocess.run(
    ["python3", "-c",
     "import pipeline.gdp_data, pipeline.feature_matrix, pipeline.prediction, "
     "pipeline.poos, pipeline.historical, pipeline.correlation_check; print('IMPORTS OK')"],
    cwd=worktree, capture_output=True, text=True,
)
print(result.stdout, result.stderr)
assert result.returncode == 0, f"import smoke test failed: {result.stderr}"

print("FULL REPO SWEEP OK")
```

- [ ] **Step 5: Re-confirm the X4 join-semantics fix survives end to end**

Re-run Task 2's `build_X4` skip-count check (it doesn't depend on anything from Tasks 3-5, but this confirms nothing in the caller migrations accidentally altered `feature_matrix.py`'s behavior):

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/feature-matrix-unification")
import pandas as pd
from pipeline.feature_matrix import build_X4

DATA_DIR = "/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data"
md = pd.read_csv(f"{DATA_DIR}/fred_md.csv")
md["sasdate"] = pd.to_datetime(md["sasdate"])
qd = pd.read_csv(f"{DATA_DIR}/fred_qd_X.csv")
gdp_df = pd.read_csv(f"{DATA_DIR}/gdp.csv")
gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
gdp = gdp_df.set_index("sasdate").sort_index()["GDPC1_t"]

X4, y4 = build_X4(md, qd, gdp, gdp, n_monthly_lags=4, n_qd_lags=4, n_gdp_lags=4)
assert X4.notna().all(axis=1).all()
print(f"X4 final shape: {X4.shape}")
print("X4 REGRESSION CHECK OK")
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Delete pipeline/output_x.py and pipeline/output_x_poos.py (fully superseded)"
```

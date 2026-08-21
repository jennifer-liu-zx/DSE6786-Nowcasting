# Close Client-Seam Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish applying the "client built once at a composition root, passed down as a parameter" pattern established in the prior Supabase-client-seam work — close the gaps that plan's own Global Constraints claimed didn't exist, but did.

**Architecture:** No new modules. Seven existing files change, all mechanically: functions that build their own Supabase client instead of accepting one gain `client` as their first parameter (or, for the two files with more than one real caller, an optional `client=None` that falls back to constructing its own). Callers thread their existing client through instead of letting callees build new ones.

**Tech Stack:** Python 3.13, supabase-py.

## Global Constraints

- `client` is the first positional parameter on every function that gains a mandatory one in this plan, matching the convention already used throughout `pipeline/poos.py`, `pipeline/dm_test.py::fetch_forecast_data`, and `pipeline/fetch_functions.py` from the prior client-seam work.
- Two functions get an *optional* `client=None` (falling back to `client or get_backend_client()`) instead of a mandatory parameter, because each genuinely has more than one real caller with different needs: `pipeline/dm_test.py::main()` (called both from its own `if __name__ == "__main__":` guard standalone, and from `pipeline/pipe.py::run()`). Every other function touched in this plan has exactly one real caller (confirmed via `grep` before writing this plan) and gets a mandatory `client` parameter — no optional fallback, no silent default construction.
- `pipeline/prediction.py::prediction_pipeline` currently builds up to 5 separate Supabase clients per call (one each from `load_filled_data`/`load_gdp`, two from `load_gdp_with_flash` since it internally calls `load_gdp` too, and one more of its own at line 324) — this plan collapses that to exactly 1.
- No pytest/unittest exists in this repo — verification is standalone `python3` scripts with `assert` statements.
- Work happens in an isolated git worktree off current `main`. Every implementer must verify `pwd`/`git rev-parse HEAD`/`git branch --show-current` before doing anything, and again before committing.
- Supabase is reachable right now with working, rotated credentials (`.env` has real `SUPABASE_URL`/`SUPABASE_ANON_KEY`/`SUPABASE_SERVICE_KEY`) — verification should include genuine live calls where practical, not just import/signature checks. `gdp`/`fred_md`/`fred_qd_x` are populated; `filled_md`/`filled_qd`/`model_forecasts`/`dm_test`/`evaluation`/`rmse` are empty (a separate backfill, out of scope) — live calls against the empty tables succeeding-but-empty is expected, not a failure.
- A Python venv with `pandas`, `numpy`, `python-dateutil`, `python-dotenv`, `supabase`, `statsmodels`, `scipy` is needed.

---

## Task 1: Fix `pipeline/evaluation_table_hist.py`'s client rebinding

**Files:**
- Modify: `pipeline/evaluation_table_hist.py:18-33` (`push_forecasts_to_evaluation`)

**Interfaces:**
- Produces: `push_forecasts_to_evaluation(client, run_date=None) -> None` — signature unchanged, but the injected `client` is now actually used for the whole function body, not just its first query.

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps")

import ast
src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps/pipeline/evaluation_table_hist.py").read()
ast.parse(src)
assert src.count("get_backend_client()") == 1, (
    f"expected exactly 1 remaining get_backend_client() call (the __main__ composition root), found {src.count('get_backend_client()')}"
)

print("ALL TASK 1 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: expected exactly 1 remaining get_backend_client() call ..., found 2`

- [ ] **Step 3: Delete the erroneous rebinding**

`push_forecasts_to_evaluation` currently uses its injected `client` parameter for its first query (checking whether any `model_forecasts` rows exist for `run_date`), then immediately discards it. Replace (currently around lines 18-33):

```python
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

    client = get_backend_client()
    df = fetch_all_model_forecasts(client)
```

with:

```python
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
```

(Just the one line, `client = get_backend_client()`, is deleted — nothing else changes. The function's later `client.table("gdp")...` call, further down in the same function, now correctly uses the originally-injected client instead of a freshly-built one.)

- [ ] **Step 4: Run the verification script again**

Expected: `ALL TASK 1 CHECKS PASSED`

- [ ] **Step 5: Live behavioral check**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps")

from database.client import get_backend_client
from pipeline.evaluation_table_hist import push_forecasts_to_evaluation

client = get_backend_client()
# model_forecasts is currently empty, so this should hit the early-return
# path cleanly (no data for any run_date) rather than erroring.
push_forecasts_to_evaluation(client, run_date="2020-01-01")
print("LIVE CHECK OK — ran against real Supabase without error")
```

- [ ] **Step 6: Commit**

```bash
git add pipeline/evaluation_table_hist.py
git commit -m "Stop evaluation_table_hist.py's push_forecasts_to_evaluation from discarding its injected client"
```

---

## Task 2: Thread `client` through `pipeline/gdp_data.py`'s three loaders

**Files:**
- Modify: `pipeline/gdp_data.py` (all three functions)

**Interfaces:**
- Produces: `load_filled_data(client) -> tuple[pd.DataFrame, pd.DataFrame]`, `load_gdp(client) -> pd.DataFrame`, `load_gdp_with_flash(client) -> pd.Series` — `client` is now each function's first (and, for `load_gdp`, only) parameter.

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps")

import ast, inspect

src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps/pipeline/gdp_data.py").read()
ast.parse(src)
assert src.count("get_backend_client()") == 0, (
    f"gdp_data.py should have zero internal client construction now — all three loaders take client as a parameter, found {src.count('get_backend_client()')}"
)

from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash
assert list(inspect.signature(load_filled_data).parameters) == ["client"]
assert list(inspect.signature(load_gdp).parameters) == ["client"]
assert list(inspect.signature(load_gdp_with_flash).parameters) == ["client"]

print("ALL TASK 2 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: gdp_data.py should have zero internal client construction now ..., found 3`

- [ ] **Step 3: Edit `pipeline/gdp_data.py`**

Replace the entire file's function bodies (keep the module docstring, imports, and `load_dotenv()` call at the top exactly as they are — only the three function definitions change):

```python
def load_filled_data(client):
    """Fetch filled_md and filled_qd from Supabase."""
    print("Loading filled data from Supabase …")

    df_md = read_table(client, "filled_md")
    df_md["sasdate"] = pd.to_datetime(df_md["sasdate"])
    df_md = df_md.sort_values("sasdate").reset_index(drop=True)
    print(f"  filled_md : {df_md.shape}")

    df_qd = read_table(client, "filled_qd")
    df_qd["sasdate"] = pd.to_datetime(df_qd["sasdate"])
    df_qd = df_qd.sort_values("sasdate").reset_index(drop=True)
    print(f"  filled_qd : {df_qd.shape}\n")

    return df_md, df_qd


def load_gdp(client) -> pd.DataFrame:
    """Fetch the raw (un-imputed) GDP table from Supabase, indexed by sasdate."""
    gdp = read_table(client, "gdp")
    gdp["sasdate"] = pd.to_datetime(gdp["sasdate"])
    gdp = gdp.set_index("sasdate").sort_index()
    gdp = gdp[gdp.index.notna()]
    return gdp


def load_gdp_with_flash(client) -> pd.Series:
    """
    Returns the GDP growth series (GDPC1_t) with any unreleased quarters
    filled by the latest Ensemble flash prediction from model_forecasts.

    Edge case: on 31 March 2026, Q4 2025 GDP may not yet be officially
    released. We substitute the Ensemble nowcast so lag features can be
    constructed for the Q1 2026 nowcast row.
    """
    gdp = load_gdp(client)
    y = gdp["GDPC1_t"].copy()

    missing = y[y.isna()].index
    if len(missing) == 0:
        return y

    for date in missing:
        resp = (client.table("model_forecasts")
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

Also remove the now-unused `from database.client import get_backend_client` import line from the top of the file (`read_table` is still imported and used; `get_backend_client` no longer is).

- [ ] **Step 4: Run the verification script again**

Expected: `ALL TASK 2 CHECKS PASSED`

- [ ] **Step 5: Live behavioral check**

`filled_md`/`filled_qd` are currently empty (a separate backfill, out of scope for this plan). Confirmed directly against this repo's unmodified `main` branch before writing this plan: `load_filled_data()` already raises `KeyError: 'sasdate'` when `filled_md` is empty — a Supabase query with zero rows returns a columnless response, and the very next line (`df_md["sasdate"] = pd.to_datetime(df_md["sasdate"])`) can't find that column. **This is pre-existing behavior, not something this task's `client`-parameter change causes.** It would crash identically with the old no-arg signature. Don't attempt to fix it here — that's a separate robustness issue, out of scope. `load_gdp`/`load_gdp_with_flash` don't have this problem since the `gdp` table has real data.

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps")

from database.client import get_backend_client
from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash

client = get_backend_client()

try:
    df_md, df_qd = load_filled_data(client)
    print(f"load_filled_data returned df_md={df_md.shape}, df_qd={df_qd.shape}")
    print("LIVE CHECK OK (filled_md apparently has data now)")
except KeyError as e:
    print(f"Got the expected pre-existing KeyError (filled_md is empty): {e}")
    print("LIVE CHECK OK — confirms client threading works; the empty-table crash is pre-existing, not introduced by this task")

gdp = load_gdp(client)
gdp_flash = load_gdp_with_flash(client)
assert gdp.shape[0] > 0, f"gdp table should have real data, got {gdp.shape[0]} rows"
assert gdp_flash.shape[0] > 0

print(f"load_gdp: {gdp.shape}")
print(f"load_gdp_with_flash: {gdp_flash.shape}")
print("LIVE CHECK OK")
```

- [ ] **Step 6: Commit**

```bash
git add pipeline/gdp_data.py
git commit -m "Thread client through gdp_data.py's three loaders instead of each constructing its own"
```

---

## Task 3: Thread `client` through `pipeline/prediction.py::prediction_pipeline`

**Files:**
- Modify: `pipeline/prediction.py:281-338` (`prediction_pipeline`)

**Interfaces:**
- Consumes: `load_filled_data(client)`, `load_gdp(client)`, `load_gdp_with_flash(client)` from `pipeline.gdp_data` (Task 2).
- Produces: `prediction_pipeline(client, run_date=None) -> None` — `client` is now the first parameter.

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps")

import ast, inspect

src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps/pipeline/prediction.py").read()
ast.parse(src)
assert "def prediction_pipeline(client" in src
assert "supabase_client = get_backend_client()" not in src, "the redundant client construction at line ~324 must be removed"
assert src.count("get_backend_client()") == 0, (
    f"prediction.py should have zero client construction now — client is always injected, found {src.count('get_backend_client()')}"
)

from pipeline.prediction import prediction_pipeline
assert list(inspect.signature(prediction_pipeline).parameters)[0] == "client"

print("ALL TASK 3 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: def prediction_pipeline(client not found`

- [ ] **Step 3: Edit `pipeline/prediction.py`'s imports**

The file currently imports `get_backend_client` for its own use (line 3: `from database.client import get_backend_client`) — check whether anything else in the file still needs it (it's used inside `nowcast_single`/`nowcast_single_latest`/`_push_to_supabase` as a parameter name, not a call — confirm via `grep -n "get_backend_client()" pipeline/prediction.py` that the only remaining call is the one this task removes). If `get_backend_client` ends up with zero remaining call sites in the file, remove the import too. Leave the import in place if anything else still calls it.

- [ ] **Step 4: Edit `prediction_pipeline`'s signature and body**

Replace the function's opening (currently lines 281-286):

```python
def prediction_pipeline(run_date=None):
    df_md, df_qd = load_filled_data()
    gdp_actual_series = load_gdp()["GDPC1_t"]
    gdp_flash_series  = load_gdp_with_flash()
    X_ar, y_ar = build_X_AR(gdp_actual_series, gdp_actual_series, n_lags=2)
```

with:

```python
def prediction_pipeline(client, run_date=None):
    df_md, df_qd = load_filled_data(client)
    gdp_actual_series = load_gdp(client)["GDPC1_t"]
    gdp_flash_series  = load_gdp_with_flash(client)
    X_ar, y_ar = build_X_AR(gdp_actual_series, gdp_actual_series, n_lags=2)
```

Then replace the redundant client construction and its four usages (currently around lines 324-338):

```python
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
```

with:

```python
    gdp_response = client.table("gdp").select("sasdate, GDPC1_t").order("sasdate", desc=False).execute()
    gdp_df = pd.DataFrame(gdp_response.data)
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    gdp_df = gdp_df.set_index("sasdate")


    quarter_dates = [
        pd.Period(gdp_df.index[-2], freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat(),
        pd.Period(gdp_df.index[-1], freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat(),
    ]

    run_all_nowcasts(gdp_df, client, run_date = run_date)
    compute_and_push_model_average(client, quarter_dates)
```

(Every `supabase_client` reference becomes `client`; the `supabase_client = get_backend_client()` construction line is deleted entirely.)

- [ ] **Step 5: Run the verification script again**

Expected: `ALL TASK 3 CHECKS PASSED`

- [ ] **Step 6: Confirm the file still parses and imports cleanly**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps")
import ast
ast.parse(open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps/pipeline/prediction.py").read())
import pipeline.prediction
print("prediction.py OK")
```

(A full live run of `prediction_pipeline` is not practical here — it depends on `filled_md`/`filled_qd` being populated, which they currently are not, a separate out-of-scope backfill. Task 4's live check on `pipe.py`'s wiring is a lighter-weight substitute; a genuine end-to-end `prediction_pipeline` run should happen once that backfill is done, outside this plan.)

- [ ] **Step 7: Commit**

```bash
git add pipeline/prediction.py
git commit -m "Thread client through prediction.py's prediction_pipeline; remove its redundant client construction"
```

---

## Task 4: Update `pipeline/pipe.py` and `pipeline/dm_test.py`

**Files:**
- Modify: `pipeline/pipe.py:30,36` (`run`'s calls to `prediction_pipeline` and `run_dm_test`)
- Modify: `pipeline/dm_test.py:274-279` (`main`)

**Interfaces:**
- Consumes: `prediction_pipeline(client, run_date=None)` from `pipeline.prediction` (Task 3).
- Produces: `main(client=None) -> None` in `pipeline/dm_test.py` — optional parameter, falls back to constructing its own client if not given (this function has two real callers: its own `__main__` guard, standalone, and `pipe.py`).

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps")

import ast, inspect

src_pipe = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps/pipeline/pipe.py").read()
ast.parse(src_pipe)
assert "prediction_pipeline(supabase, run_date=" in src_pipe
assert "run_dm_test(client=supabase)" in src_pipe

src_dm = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps/pipeline/dm_test.py").read()
ast.parse(src_dm)
assert "def main(client=None):" in src_dm

from pipeline.dm_test import main
sig = inspect.signature(main)
assert list(sig.parameters) == ["client"]
assert sig.parameters["client"].default is None

print("ALL TASK 4 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: prediction_pipeline(supabase, run_date= not found`

- [ ] **Step 3: Edit `pipeline/pipe.py`'s `run()`**

Replace (currently lines 30 and 36 — not adjacent, edit each independently):

```python
    prediction_pipeline(run_date=pd.to_datetime(run_date) if run_date else None)
```
with:
```python
    prediction_pipeline(supabase, run_date=pd.to_datetime(run_date) if run_date else None)
```

and:

```python
    run_dm_test()
```
with:
```python
    run_dm_test(client=supabase)
```

(`supabase` is already constructed at line 19, earlier in the same `run()` function — nothing else about `pipe.py` changes.)

- [ ] **Step 4: Edit `pipeline/dm_test.py`'s `main()`**

Replace (currently around lines 275-277):

```python
def main():
    # Initialize Supabase Client
    supabase = get_backend_client()
```

with:

```python
def main(client=None):
    # Initialize Supabase Client — accept an injected one (e.g. from pipe.py,
    # which already has one), or build our own for standalone runs.
    supabase = client or get_backend_client()
```

Nothing else in `main()`'s body changes — it already refers to the local variable `supabase` throughout (in its call to `fetch_forecast_data(supabase)` and `push_dm_results_to_supabase(client=supabase, ...)`), which still works unchanged since `supabase` is still the name bound inside the function.

- [ ] **Step 5: Run the verification script again**

Expected: `ALL TASK 4 CHECKS PASSED`

- [ ] **Step 6: Live behavioral check — confirm both call shapes work**

`main()` calls `fetch_forecast_data(supabase)` internally, and the `evaluation` table is currently empty. This is the exact same pre-existing bug already documented and worked around in the prior client-seam plan (`docs/superpowers/plans/2026-08-20-supabase-client-seam.md`, Task 2): `fetch_forecast_data` raises `KeyError` on an empty `evaluation` table because a zero-row query result has no columns for `.melt()` to find — confirmed there against this repo's original code, unrelated to any client-parameter change. `main()` will hit this same `KeyError` regardless of which call shape is used. This step confirms both call shapes reach that same, already-understood failure point (not a new one), rather than expecting a clean success:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps")

from database.client import get_backend_client
from pipeline.dm_test import main

client = get_backend_client()

# Shape 1: called with an injected client (how pipe.py calls it now)
try:
    main(client=client)
    print("Shape 1 (injected client): succeeded (evaluation table apparently has data now)")
except KeyError as e:
    print(f"Shape 1 (injected client): hit the expected pre-existing KeyError: {str(e)[:100]}")

# Shape 2: called with no arguments (how it's run standalone) -- falls back
# to constructing its own client internally.
try:
    main()
    print("Shape 2 (no args, fallback client): succeeded (evaluation table apparently has data now)")
except KeyError as e:
    print(f"Shape 2 (no args, fallback client): hit the expected pre-existing KeyError: {str(e)[:100]}")

print("LIVE CHECK OK — both call shapes reach real Supabase and fail (or succeed) identically; no new failure mode introduced by the client-parameter change")
```

- [ ] **Step 7: Commit**

```bash
git add pipeline/pipe.py pipeline/dm_test.py
git commit -m "Thread pipe.py's client into prediction_pipeline and dm_test.main; give main() an optional client parameter"
```

---

## Task 5: Fix `pipeline/correlation_check.py` and `pipeline/feature_matrix.py`'s `__main__` block

**Files:**
- Modify: `pipeline/correlation_check.py:47-49` (module-level loader calls)
- Modify: `pipeline/feature_matrix.py:341-348` (`__main__` block's loader calls)

**Interfaces:**
- Consumes: `load_filled_data(client)`, `load_gdp(client)`, `load_gdp_with_flash(client)` from `pipeline.gdp_data` (Task 2).

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps")

import ast

src_cc = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps/pipeline/correlation_check.py").read()
ast.parse(src_cc)
assert "load_filled_data(client)" in src_cc
assert 'load_gdp(client)["GDPC1_t"]' in src_cc
assert "load_gdp_with_flash(client)" in src_cc

src_fm = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps/pipeline/feature_matrix.py").read()
ast.parse(src_fm)
assert "load_filled_data(client)" in src_fm
assert 'load_gdp(client)["GDPC1_t"]' in src_fm
assert "load_gdp_with_flash(client)" in src_fm

print("ALL TASK 5 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError` (both files still call the loaders with no `client` argument)

- [ ] **Step 3: Edit `pipeline/correlation_check.py`**

Add an import near the existing `from pipeline.gdp_data import ...` line (currently around line 30):
```python
from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash
from database.client import get_backend_client
```

Replace the loader-calling block (currently around lines 47-49):
```python
df_md, df_qd = load_filled_data()
gdp_actual_series = load_gdp()["GDPC1_t"]
gdp_flash_series  = load_gdp_with_flash()
```
with:
```python
client = get_backend_client()
df_md, df_qd = load_filled_data(client)
gdp_actual_series = load_gdp(client)["GDPC1_t"]
gdp_flash_series  = load_gdp_with_flash(client)
```

- [ ] **Step 4: Edit `pipeline/feature_matrix.py`'s `__main__` block**

Replace (currently around lines 341-348):
```python
if __name__ == "__main__":
    from pathlib import Path
    from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_DIR / "data"

    df_md, df_qd = load_filled_data()
    gdp_actual_series = load_gdp()["GDPC1_t"]
    gdp_flash_series  = load_gdp_with_flash()
```
with:
```python
if __name__ == "__main__":
    from pathlib import Path
    from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash
    from database.client import get_backend_client

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_DIR / "data"

    client = get_backend_client()
    df_md, df_qd = load_filled_data(client)
    gdp_actual_series = load_gdp(client)["GDPC1_t"]
    gdp_flash_series  = load_gdp_with_flash(client)
```

- [ ] **Step 5: Run the verification script again**

Expected: `ALL TASK 5 CHECKS PASSED`

- [ ] **Step 6: Live behavioral check — confirm `correlation_check.py`'s import chain still resolves correctly**

`correlation_check.py` is a script that runs its analysis at module scope (no `__main__` guard around most of it), and it already needed live Supabase to import before this change (confirmed during earlier session work) — this check confirms the specific edit didn't introduce a *new* kind of failure (e.g. a broken reference), only that the existing credential-dependent behavior is unchanged:

```python
import subprocess
result = subprocess.run(
    ["python3", "-c", "import pipeline.correlation_check"],
    cwd="/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps",
    capture_output=True, text=True, timeout=60,
)
print(result.stdout[-1000:])
print(result.stderr[-1000:])
# Supabase is reachable right now, so this should actually succeed (unlike
# the credential-gap case from the prior client-seam plan) -- but if it
# fails, it must fail for a data/analysis reason (e.g. empty filled_md
# breaking a downstream calculation), not an ImportError/AttributeError/
# TypeError, which would indicate a broken reference from this edit.
if result.returncode != 0:
    assert "ImportError" not in result.stderr and "AttributeError" not in result.stderr and "TypeError" not in result.stderr, (
        f"correlation_check.py failed with a broken-reference-style error, not a data error: {result.stderr[-500:]}"
    )
    print("Non-zero exit, but not from a broken reference -- likely a downstream data issue given filled_md/filled_qd are still empty. Acceptable for this task.")
else:
    print("pipeline.correlation_check imported and ran to completion.")
print("LIVE CHECK OK")
```

- [ ] **Step 7: Confirm `feature_matrix.py`'s `__main__` block still runs (same style of check as the prior client-seam plan)**

```python
import subprocess, os
stripped_env = {k: v for k, v in os.environ.items() if not k.startswith("SUPABASE")}
result = subprocess.run(
    ["python3", "-c", "import pipeline.feature_matrix; print('IMPORTED OK')"],
    cwd="/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps",
    capture_output=True, text=True, env=stripped_env,
)
print(result.stdout, result.stderr)
assert result.returncode == 0, f"pipeline.feature_matrix should still import cleanly (it's a pure module -- the __main__ block only runs when invoked directly): {result.stderr}"
assert "IMPORTED OK" in result.stdout
print("FEATURE_MATRIX IMPORT CHECK OK")
```

- [ ] **Step 8: Commit**

```bash
git add pipeline/correlation_check.py pipeline/feature_matrix.py
git commit -m "Thread a single client through correlation_check.py and feature_matrix.py's __main__ block"
```

---

## Task 6: Full-repo consistency sweep

**Files:**
- None modified — verification only.

**Interfaces:**
- None — this task confirms the prior five tasks compose correctly.

- [ ] **Step 1: Write and run the sweep script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps")

import ast, subprocess

worktree = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps"

files = [
    "pipeline/evaluation_table_hist.py", "pipeline/gdp_data.py", "pipeline/prediction.py",
    "pipeline/pipe.py", "pipeline/dm_test.py", "pipeline/correlation_check.py",
    "pipeline/feature_matrix.py",
]
for f in files:
    src = open(f"{worktree}/{f}").read()
    ast.parse(src)
    print(f, "parses OK")

# Count every remaining get_backend_client()/get_frontend_client() call across
# the whole pipeline/ tree -- each one should be a genuine composition root.
# Expected remaining call sites after this plan:
#   pipeline/evaluation_table_hist.py -- 1 (its __main__ block)
#   pipeline/gdp_data.py              -- 0
#   pipeline/prediction.py            -- 0 (or import removed entirely)
#   pipeline/pipe.py                  -- 1 (run()'s own composition root)
#   pipeline/dm_test.py               -- 1 (main()'s fallback + its own __main__)
#   pipeline/correlation_check.py     -- 1 (its own module-level composition root)
#   pipeline/feature_matrix.py        -- 1 (its own __main__ block)
result = subprocess.run(
    ["grep", "-c", "get_backend_client()"] + [f"{worktree}/{f}" for f in files],
    capture_output=True, text=True,
)
print(result.stdout)

# Import smoke test for everything that's genuinely import-safe without credentials
result2 = subprocess.run(
    ["python3", "-c",
     "import pipeline.evaluation_table_hist, pipeline.gdp_data, pipeline.prediction, "
     "pipeline.pipe, pipeline.dm_test, pipeline.feature_matrix; print('IMPORTS OK')"],
    cwd=worktree, capture_output=True, text=True,
)
print(result2.stdout, result2.stderr)
assert result2.returncode == 0, f"import smoke test failed: {result2.stderr}"

print("FULL REPO SWEEP OK")
```

Read the grep output and confirm each file's count matches the expected list in the comment above — if any file has more or fewer `get_backend_client()` calls than expected, that's a real gap this plan didn't fully close; investigate before proceeding (don't just note it and move on).

- [ ] **Step 2: Live end-to-end check — `pipe.py`'s wiring, without running the full monthly pipeline**

`pipe.py::run()` has a hard guard that no-ops unless run on the last calendar day of the month, and running it for real would also trigger `load_main()` (a live FRED fetch) and the full prediction/evaluation chain — too heavy for a verification step. Instead, confirm the specific wiring fixed by Task 4 directly. As in Task 4 Step 6, `run_dm_test(client=...)` will hit the same pre-existing, already-documented `KeyError` from `fetch_forecast_data` on the empty `evaluation` table — expected, not a failure:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/close-client-seam-gaps")

from database.client import get_backend_client
from pipeline.dm_test import main as run_dm_test
from pipeline.prediction import prediction_pipeline
import inspect

client = get_backend_client()

# Confirm prediction_pipeline's signature genuinely requires client first --
# this is what pipe.py now passes its own `supabase` variable into.
assert list(inspect.signature(prediction_pipeline).parameters)[0] == "client"

# Confirm dm_test.main's optional-client shape works when called the way
# pipe.py calls it now (expect the same pre-existing empty-evaluation-table
# KeyError documented in Task 4 Step 6 — this call reaching that point at
# all, rather than an earlier TypeError/AttributeError, is what confirms
# the wiring itself is correct).
try:
    run_dm_test(client=client)
    print("run_dm_test(client=...) succeeded (evaluation table apparently has data now)")
except KeyError as e:
    print(f"run_dm_test(client=...) hit the expected pre-existing KeyError: {str(e)[:100]}")

print("LIVE CHECK OK — pipe.py's wiring changes both verified against real signatures/calls")
```

- [ ] **Step 3: Commit**

Only if the sweep found something to fix. If the sweep is clean, no commit is needed for this task — just report the clean sweep result.

If a fix was needed:
```bash
git add -A
git commit -m "Fix stragglers found in the full-repo client-seam consistency sweep"
```

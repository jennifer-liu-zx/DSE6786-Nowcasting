# Supabase Client Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Supabase dependency a real seam — every function that talks to Supabase accepts a client as a parameter instead of constructing (or reaching for) its own — closing the three places in this repo that still violate that pattern.

**Architecture:** No new module. `database/client.py` gains a second constructor, `get_frontend_client()` (mirrors the existing `get_backend_client()`, reads `SUPABASE_ANON_KEY` instead of `SUPABASE_SERVICE_KEY`). Three existing call sites move from "construct my own client" to "accept a client parameter": `pipeline/poos.py`'s `cut_and_fill`/`poos_validation` (currently reconstructs a client on every one of up to 100 loop iterations), `pipeline/dm_test.py`'s `fetch_forecast_data` (constructs its own, called with no arguments), and every function in `pipeline/fetch_functions.py` (currently a module-level singleton client built at import time — the reason merely importing `app.py` requires live credentials). `app.py` builds its client once, inside `server(input, output, session)`, not at module scope.

**Tech Stack:** Python 3.13, supabase-py, python-dotenv.

## Global Constraints

- Everything else in the codebase already follows the correct pattern — `evaluation_table_hist.py`, `prediction.py`, `historical.py`'s other functions, `ci_update.py`, `plot_poos.py` all already accept `client` as a parameter in their worker functions, constructing it exactly once at their own entry point (`run()`/`main()`/`*_pipeline()`/`if __name__ == "__main__":`). Do not touch any of those files — this plan's scope is the three genuine violations listed above.
- `client` is the first positional parameter on every function that gains one in this plan, matching the convention already used by `push_poos_to_supabase(client, ...)` and `push_evaluation_to_supabase(client, ...)` in `pipeline/historical.py`.
- No fake-client test infrastructure is in scope. The parameter-passing seam is the entire deliverable; a fake/in-memory adapter for tests is a capability this unlocks later, not something to build now.
- `pipeline/fetch_functions.py`'s `fetch_nowcast_x_labels` function has already been deleted in the working tree (confirmed dead: zero callers anywhere in the repo, and its one piece of logic was already duplicated inside `fetch_nowcast_data`). This plan's Task 3 folds that pre-made edit into its own commit rather than re-doing it — the implementer should confirm the deletion is already present, not re-delete it.
- Supabase is reachable today and `gdp`/`fred_md`/`fred_qd_x` are populated (267/807/267 rows) — every table in `database/schema.sql` has an `anon`-role read policy, confirmed directly: `client.table("gdp").select(...)` with the `SUPABASE_ANON_KEY` client returns real rows. This means Task 3 and Task 4's verification can be genuine live Supabase calls, not just import/signature checks like the prior two candidates in this session had to settle for. `filled_md`, `filled_qd`, `model_forecasts`, `dm_test`, `evaluation`, and `rmse` are still empty (a separate, larger backfill out of scope here) — live queries against those tables will succeed but return empty results; that's expected, not a failure.
- No pytest/unittest exists in this repo — verification is standalone `python3` scripts with `assert` statements, matching the existing `if __name__ == "__main__":` idiom.
- Work happens in an isolated git worktree off current `main`. Every implementer must verify `pwd`/`git rev-parse HEAD`/`git branch --show-current` before doing anything, and again before committing.
- Any verification script using `sys.path.insert(...)` must point at the worktree checkout path, not the main repo path.
- A Python venv with `pandas`, `numpy`, `python-dateutil`, `python-dotenv`, `supabase`, `statsmodels`, `certifi` is needed. If SSL certificate errors occur on any live network call (a known issue with fresh venvs on this machine), set `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` to `python3 -c "import certifi; print(certifi.where())"`'s output before running.
- `.env` must be copied into the worktree (gitignored, not tracked) — it now has `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `SUPABASE_SERVICE_KEY` all set.

---

## Task 1: Thread `client` through `pipeline/poos.py` and its caller in `pipeline/historical.py`

**Files:**
- Modify: `pipeline/poos.py:83-153` (`cut_and_fill`), `pipeline/poos.py:185-225` (`poos_validation`), `pipeline/poos.py:357-380` (`__main__` smoke test)
- Modify: `pipeline/historical.py:145-165` (`poos_validation` call site)

**Interfaces:**
- Produces: `cut_and_fill(client, version, q_predicted, QD_t, MD_t, gdp, model_name="All_Model_Average")` (client is now the first parameter; everything else unchanged). `poos_validation(client, method, buildname, QD_t, MD_t, y_full, version, num_test=TEST_SIZE, num_train=TRAIN_SIZE)` (same insertion).

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam")

import ast
import inspect

src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam/pipeline/poos.py").read()
ast.parse(src)
assert "def cut_and_fill(client" in src, "cut_and_fill must take client as its first parameter"
assert "def poos_validation(client" in src, "poos_validation must take client as its first parameter"
assert src.count("get_backend_client()") == 1, (
    f"expected exactly 1 remaining get_backend_client() call (the __main__ smoke test), found {src.count('get_backend_client()')}"
)

from pipeline.poos import cut_and_fill, poos_validation
assert list(inspect.signature(cut_and_fill).parameters)[0] == "client"
assert list(inspect.signature(poos_validation).parameters)[0] == "client"

src_hist = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam/pipeline/historical.py").read()
ast.parse(src_hist)
assert "poos_validation(\n                client=client," in src_hist or "poos_validation(client, " in src_hist or "poos_validation(client=client" in src_hist, (
    "historical.py's poos_validation call must pass its existing client through"
)

print("ALL TASK 1 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: cut_and_fill must take client as its first parameter`

- [ ] **Step 3: Edit `pipeline/poos.py` — `cut_and_fill`**

Replace the function signature and its first body line (currently lines 83-91):
```python
def cut_and_fill(version: int,
                 q_predicted: pd.Timestamp,
                 QD_t: pd.DataFrame,
                 MD_t: pd.DataFrame,
                 gdp: pd.Series,
                 model_name: str = "All_Model_Average",
                 ):

    client = get_backend_client()   
```
with:
```python
def cut_and_fill(client,
                 version: int,
                 q_predicted: pd.Timestamp,
                 QD_t: pd.DataFrame,
                 MD_t: pd.DataFrame,
                 gdp: pd.Series,
                 model_name: str = "All_Model_Average",
                 ):
```
(the rest of the function body is unchanged — it already references `client.table("model_forecasts")` further down; only the parameter list and the now-deleted construction line change.)

- [ ] **Step 4: Edit `pipeline/poos.py` — `poos_validation`**

Replace the function signature (currently lines 185-193):
```python
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
```
with:
```python
def poos_validation(
    client,
    method: Callable,
    buildname: str,
    QD_t: pd.DataFrame,
    MD_t: pd.DataFrame,
    y_full: pd.Series,
    version: int,
    num_test: int = TEST_SIZE,
    num_train: int = TRAIN_SIZE,
) -> Tuple[pd.DataFrame, float, float]:
```

Then, in its body, replace the `cut_and_fill` call (currently around line 217-223):
```python
        # 1. Cut & fill
        qd_filled, md_filled, gdp_filled, gdp_raw = cut_and_fill(
            version=version,
            q_predicted=pd.Timestamp(q_predicted),
            QD_t=QD_t,
            MD_t=MD_t,
            gdp=y_full
        )
```
with:
```python
        # 1. Cut & fill
        qd_filled, md_filled, gdp_filled, gdp_raw = cut_and_fill(
            client,
            version=version,
            q_predicted=pd.Timestamp(q_predicted),
            QD_t=QD_t,
            MD_t=MD_t,
            gdp=y_full
        )
```

- [ ] **Step 5: Edit `pipeline/poos.py` — `__main__` smoke test**

This is the one remaining legitimate `get_backend_client()` call — it becomes the composition root for the standalone smoke test. Replace (currently around lines 357-376):
```python
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
```
with:
```python
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

    smoke_test_client = get_backend_client()

    # Smoke-test cut_and_fill
    filled_qd, filled_md, gdp_filled, gdp_raw = cut_and_fill(
        smoke_test_client,
        version=4,
        q_predicted=pd.Timestamp("2025-12-01"),
        QD_t=qd,
        MD_t=md,
        gdp=gdp_df["GDPC1_t"]
    )
```

Leave the rest of the `__main__` block (the `make_build_X("X1")` smoke test further down) untouched — it doesn't call `cut_and_fill` or need a client.

- [ ] **Step 6: Edit `pipeline/historical.py`'s `poos_validation` call site**

Replace (currently around lines 155-162, inside `run()`, which already has `client = get_backend_client()` earlier in the same function):
```python
            poos_out, rmse, mae = poos_validation(
                method=cfg["model"],
                buildname=build_name,
                QD_t=QD_t,
                MD_t=MD_t,
                y_full=y_full,
                version=version,
            )
```
with:
```python
            poos_out, rmse, mae = poos_validation(
                client,
                method=cfg["model"],
                buildname=build_name,
                QD_t=QD_t,
                MD_t=MD_t,
                y_full=y_full,
                version=version,
            )
```

- [ ] **Step 7: Run the verification script again**

Expected: `ALL TASK 1 CHECKS PASSED`

- [ ] **Step 8: Live behavioral check — confirm the smoke test still runs correctly**

Supabase is reachable and `gdp`/`fred_md`/`fred_qd_x` are populated, so this can be a real run rather than an import-only check:

```python
import subprocess
result = subprocess.run(
    ["python3", "pipeline/poos.py"],
    cwd="/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam",
    capture_output=True, text=True, timeout=120,
)
print(result.stdout[-2000:])
print(result.stderr[-2000:])
assert result.returncode == 0, f"poos.py __main__ smoke test failed: {result.stderr}"
assert "Feature matrix tail:" in result.stdout
print("POOS SMOKE TEST OK (live Supabase)")
```

- [ ] **Step 9: Commit**

```bash
git add pipeline/poos.py pipeline/historical.py
git commit -m "Thread client through poos.py's cut_and_fill/poos_validation instead of constructing one per loop iteration"
```

---

## Task 2: Thread `client` through `pipeline/dm_test.py`'s `fetch_forecast_data`

**Files:**
- Modify: `pipeline/dm_test.py:216-238` (`fetch_forecast_data`), `pipeline/dm_test.py:275-310` (`main`, its call site), `pipeline/dm_test.py:314-320` (dead commented-out block)

**Interfaces:**
- Produces: `fetch_forecast_data(client, table_name: str = "evaluation") -> pd.DataFrame` (client is now the first parameter).

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam")

import ast, inspect

src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam/pipeline/dm_test.py").read()
ast.parse(src)
assert "def fetch_forecast_data(client" in src
assert src.count("get_backend_client()") == 1, (
    f"expected exactly 1 remaining get_backend_client() call (main()'s composition root), found {src.count('get_backend_client()')}"
)
assert "# df_forecasts = fetch_forecast_data()" not in src, "stale commented-out block referencing the old no-arg signature should be removed"

from pipeline.dm_test import fetch_forecast_data
assert list(inspect.signature(fetch_forecast_data).parameters)[0] == "client"

print("ALL TASK 2 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: def fetch_forecast_data(client not found`

- [ ] **Step 3: Edit `pipeline/dm_test.py` — `fetch_forecast_data`**

Replace (currently lines 216-224):
```python
def fetch_forecast_data(table_name: str = "evaluation") -> pd.DataFrame:
    """
    Pulls all forecast data from Supabase and returns a cleaned DataFrame.
    """
    supabase = get_backend_client()
    response = supabase.table(table_name).select("quarter_date",
```
with:
```python
def fetch_forecast_data(client, table_name: str = "evaluation") -> pd.DataFrame:
    """
    Pulls all forecast data from Supabase and returns a cleaned DataFrame.
    """
    response = client.table(table_name).select("quarter_date",
```

- [ ] **Step 4: Edit `pipeline/dm_test.py` — `main()`'s call site**

Replace (currently around line 281):
```python
    df_forecasts = fetch_forecast_data()
```
with:
```python
    df_forecasts = fetch_forecast_data(supabase)
```
(`supabase` is already constructed two lines earlier in `main()` via `supabase = get_backend_client()` — this is the composition root, unchanged.)

- [ ] **Step 5: Delete the stale commented-out test block**

Delete this block entirely (currently around lines 314-320, right after `if __name__ == "__main__": main()`):
```python
# ── TEST ──────────────────────────────────────────────────────────────────────
# df_forecasts = fetch_forecast_data()
# print(df_forecasts.head(15))
# model_pairs = compare_model_pairs(
#     df_forecasts,
#     time_col='quarter_date'
```
It references the old no-arg `fetch_forecast_data()` signature and was already dead (commented out, incomplete — the block is cut off mid-call in the current file). Removing it avoids leaving a stale, now-actively-wrong example in the file.

- [ ] **Step 6: Run the verification script again**

Expected: `ALL TASK 2 CHECKS PASSED`

- [ ] **Step 7: Live behavioral check**

`evaluation` is currently empty (0 rows). Confirmed directly against this repo's unmodified `main` branch before writing this plan: `fetch_forecast_data()` already raises `KeyError` on an empty `evaluation` table — `pd.DataFrame(response.data)` on zero rows produces a columnless DataFrame, and the function's `.melt(id_vars=[...])` call then can't find the columns it expects. **This is pre-existing behavior, not something this task's `client`-parameter change causes or is responsible for fixing** — it would crash identically with the old no-arg signature. Don't attempt to fix it here; that's a separate robustness issue, out of scope for a dependency-injection refactor. This step just confirms the client-threading itself is what's live-tested, by checking the failure mode is exactly the expected pre-existing one and nothing new (an `ImportError`/`AttributeError`/`TypeError` here would mean the refactor broke something):

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam")

from database.client import get_backend_client
from pipeline.dm_test import fetch_forecast_data

client = get_backend_client()
try:
    df = fetch_forecast_data(client)
    print(f"fetch_forecast_data returned {len(df)} rows with columns {list(df.columns)}")
    print("LIVE CHECK OK (evaluation table apparently has data now)")
except KeyError as e:
    print(f"Got the expected pre-existing KeyError (evaluation table is empty): {str(e)[:150]}")
    print("LIVE CHECK OK — confirms client threading works; the empty-table crash is pre-existing, not introduced by this task")
```

- [ ] **Step 8: Commit**

```bash
git add pipeline/dm_test.py
git commit -m "Thread client through dm_test.py's fetch_forecast_data; drop stale commented-out block"
```

---

## Task 3: Add `get_frontend_client()`; convert `pipeline/fetch_functions.py` off its module-level singleton

**Files:**
- Modify: `database/client.py` (add `get_frontend_client`)
- Modify: `pipeline/fetch_functions.py` (remove module-level client, add `client` parameter to every Supabase-touching function)

**Interfaces:**
- Produces: `get_frontend_client() -> Client` in `database/client.py`, reads `SUPABASE_ANON_KEY`. `fetch_nowcast_data(client, quarter)`, `fetch_confidence_intervals(client, quarter, model)`, `fetch_flash_predictions(client, start_date, end_date, flash_month)`, `fetch_historical_data(client, start_date, end_date, flash_month)`, `fetch_rmse(client, models)`, `fetch_dm(client, models, flash_month)`, `fetch_realised_gdp(client, quarter)` — `client` is the first parameter on all seven.

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam")

import ast, inspect

src_client = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam/database/client.py").read()
ast.parse(src_client)
assert "def get_frontend_client(" in src_client

src_ff = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam/pipeline/fetch_functions.py").read()
ast.parse(src_ff)
assert "create_client(" not in src_ff, "module-level client construction must be gone"
assert "fetch_nowcast_x_labels" not in src_ff, "already-deleted dead function should stay deleted"

from database.client import get_frontend_client
from pipeline.fetch_functions import (
    fetch_nowcast_data, fetch_confidence_intervals, fetch_flash_predictions,
    fetch_historical_data, fetch_rmse, fetch_dm, fetch_realised_gdp,
)
for fn in [fetch_nowcast_data, fetch_confidence_intervals, fetch_flash_predictions,
           fetch_historical_data, fetch_rmse, fetch_dm, fetch_realised_gdp]:
    params = list(inspect.signature(fn).parameters)
    assert params[0] == "client", f"{fn.__name__} must take client as its first parameter, got {params}"

print("ALL TASK 3 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: def get_frontend_client( not found` (or similar, on the first assertion)

- [ ] **Step 3: Add `get_frontend_client()` to `database/client.py`**

Append this function after the existing `get_backend_client()` (the full file becomes):
```python
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def get_backend_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url:
        raise EnvironmentError("SUPABASE_URL not found in environment variables.")
    if not key:
        raise EnvironmentError("SUPABASE_SERVICE_KEY not found in environment variables.")

    return create_client(url, key)


def get_frontend_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url:
        raise EnvironmentError("SUPABASE_URL not found in environment variables.")
    if not key:
        raise EnvironmentError("SUPABASE_ANON_KEY not found in environment variables.")

    return create_client(url, key)
```

- [ ] **Step 4: Rewrite `pipeline/fetch_functions.py`**

First, confirm the already-made deletion of `fetch_nowcast_x_labels` is present (it should already be gone between `fetch_nowcast_data` and the `Function 3` comment — if it's somehow still there, delete it now: it's confirmed dead, zero callers anywhere in the repo).

Replace the top of the file (currently lines 1-16):
```python
from supabase import create_client
from dotenv import load_dotenv
import os
import calendar
from datetime import date
from itertools import combinations

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")

assert url is not None, "SUPABASE_URL environment variable is not set"
assert key is not None, "SUPABASE_ANON_KEY environment variable is not set"

supabase = create_client(url, key)
```
with:
```python
import calendar
from datetime import date
from itertools import combinations
```
(`os`, `create_client`, and `load_dotenv` are no longer needed in this file at all — client construction now lives entirely in `database/client.py`.)

Then replace each function body's `supabase.table(...)` references with `client.table(...)`, and add `client` as each function's first parameter. The full set of changes:

`fetch_nowcast_data`:
```python
def fetch_nowcast_data(client, quarter: str) -> dict[str, list[float]]:
    
    # Step 0: Get input's (quarter) start and end dates:
    quarter_start = quarter_to_dates(quarter)
    
    # Step 1: Get rows from model_forecasts table for the specified quarter
    result = client.table("model_forecasts") \
        .select("*") \
        .eq("quarter_date", quarter_start) \
        .order("month_date") \
        .execute()
```
(the rest of the function body is unchanged)

`fetch_confidence_intervals`:
```python
def fetch_confidence_intervals(client, quarter: str, model: str) -> tuple[list[str], list[float], list[float]]:
    
    # Step 0: Get input's (quarter) start and end dates:
    quarter_start = quarter_to_dates(quarter)
    
    # Step 1: Repeat step 1; query supabase
    result = client.table("model_forecasts") \
    .select("month_date", "ci_50_lb", "ci_50_ub", "ci_80_lb", "ci_80_ub") \
    .eq("model_name", model) \
    .eq("quarter_date", quarter_start) \
    .order("month_date") \
    .execute()
```
(the rest of the function body is unchanged)

`fetch_flash_predictions` (note: this one is also called internally by `fetch_historical_data`, so both signature and internal query change):
```python
def fetch_flash_predictions(
    client, start_date, end_date, flash_month: int
) -> dict[str, list[float]]:
 
    pairs = _flash_month_dates(start_date, end_date, flash_month)
    if not pairs:
        return {}
 
    predictions: dict[str, list[float]] = {}
 
    for quarter_date, month_date in pairs:
        result = client.table("model_forecasts") \
            .select("model_name, nowcast") \
            .eq("quarter_date", quarter_date) \
            .eq("month_date", month_date) \
            .execute()
 
        for row in result.data:
            model = row["model_name"]
            if model not in predictions:
                predictions[model] = []
            predictions[model].append(row["nowcast"])
 
    return predictions
```

`fetch_historical_data` (both its own `supabase.table("gdp")` call and its internal call to `fetch_flash_predictions` need `client` threaded through):
```python
def fetch_historical_data(
    client, start_date, end_date, flash_month: int
) -> tuple[list[str], list[float], dict[str, list[float]]]:
 
    # Actual GDP values — one row per quarter, date is the quarter start
    actuals_results = client.table("gdp") \
        .select("sasdate", "GDPC1_t") \
        .gte("sasdate", str(start_date)) \
        .lte("sasdate", str(end_date)) \
        .order("sasdate") \
        .execute()
 
    actuals_rows   = actuals_results.data
    quarter_labels = [row["sasdate"]  for row in actuals_rows]
    actual_values  = [row["GDPC1_t"] for row in actuals_rows]
 
    # Model predictions: one value per quarter, from the chosen flash month
    predictions = fetch_flash_predictions(client, start_date, end_date, flash_month)
 
    return quarter_labels, actual_values, predictions
```

`fetch_rmse`:
```python
def fetch_rmse(client, models: list[str]) -> dict[str, dict]:
 
    result = client.table("rmse") \
        .select("model", "version", "rmse") \
        .in_("model", models) \
        .execute()
```
(the rest of the function body is unchanged)

`fetch_dm` (two `supabase.table("dm_test")` calls inside the loop, both become `client.table(...)`):
```python
def fetch_dm(client, models: list[str], flash_month: int) -> list[dict]:
    """
    Returns one row per unique pair of models: {model_1, model_2,
    test_statistic, p_value}. Uses whichever (model_1, model_2) ordering
    the dm_test table stores for that pair — winner-first, i.e. a negative
    test_statistic always favours model_1.
    """
    results = []
    for m1, m2 in combinations(models, 2):
        result = client.table("dm_test") \
            .select("model_1", "model_2", "test_statistic", "p_value") \
            .eq("model_1", m1).eq("model_2", m2).eq("version", flash_month) \
            .execute()
        if not result.data:
            result = client.table("dm_test") \
                .select("model_1", "model_2", "test_statistic", "p_value") \
                .eq("model_1", m2).eq("model_2", m1).eq("version", flash_month) \
                .execute()
        if result.data:
            row = result.data[0]
            results.append({
                "model_1": row["model_1"],
                "model_2": row["model_2"],
                "test_statistic": row["test_statistic"],
                "p_value": row["p_value"],
            })
    return results
```

`fetch_realised_gdp`:
```python
def fetch_realised_gdp(client, quarter: str) -> float | None:
    quarter_start = quarter_to_dates(quarter)
    result = client.table("gdp") \
        .select("GDPC1_t") \
        .eq("sasdate", quarter_start) \
        .execute()
    if result.data:
        return result.data[0]["GDPC1_t"]
    return None
```

`quarter_to_dates`, `_month_end`, and `_flash_month_dates` are pure helpers with no Supabase access — leave them exactly as they are.

- [ ] **Step 5: Run the verification script again**

Expected: `ALL TASK 3 CHECKS PASSED`

- [ ] **Step 6: Live behavioral check — confirm `fetch_realised_gdp` returns a real value**

`gdp` is populated with real data now, so this is a genuine end-to-end check, not just a signature check:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam")

from database.client import get_frontend_client
from pipeline.fetch_functions import fetch_realised_gdp, fetch_rmse, fetch_dm

client = get_frontend_client()

# fetch_realised_gdp: gdp table has real rows now — 2024:Q1 should return a real float
val = fetch_realised_gdp(client, "2024:Q1")
assert val is not None, "expected a real GDP value for 2024:Q1"
assert isinstance(val, float)
print(f"fetch_realised_gdp(client, '2024:Q1') = {val}")

# fetch_rmse / fetch_dm: rmse/dm_test tables are empty — should return empty results, not error
rmse = fetch_rmse(client, ["AR_Benchmark"])
assert rmse == {}, f"expected empty dict (rmse table not backfilled yet), got {rmse}"
dm = fetch_dm(client, ["AR_Benchmark", "LASSO_Average"], flash_month=1)
assert dm == [], f"expected empty list (dm_test table not backfilled yet), got {dm}"

print("LIVE CHECK OK — importing this module no longer requires credentials, and calling its functions against live Supabase works")
```

- [ ] **Step 7: Confirm the module is importable with zero credentials present (the actual bug this fixes)**

```python
import subprocess
result = subprocess.run(
    ["python3", "-c", "import pipeline.fetch_functions; print('IMPORTED WITH NO ENV VARS')"],
    cwd="/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam",
    capture_output=True, text=True,
    env={},  # deliberately empty environment — no SUPABASE_URL/ANON_KEY/SERVICE_KEY, no PATH even
)
```
This particular subprocess call needs a minimally-populated `env` (an entirely empty environment will likely fail to even find `python3` on `PATH`) — use this instead:
```python
import subprocess, os
stripped_env = {k: v for k, v in os.environ.items() if not k.startswith("SUPABASE")}
result = subprocess.run(
    ["python3", "-c", "import pipeline.fetch_functions; print('IMPORTED WITH NO SUPABASE ENV VARS')"],
    cwd="/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam",
    capture_output=True, text=True, env=stripped_env,
)
print(result.stdout, result.stderr)
assert result.returncode == 0, f"pipeline.fetch_functions should import cleanly with no Supabase credentials at all: {result.stderr}"
assert "IMPORTED WITH NO SUPABASE ENV VARS" in result.stdout
print("IMPORT-WITHOUT-CREDENTIALS CHECK OK — this is the bug this task actually fixes")
```

- [ ] **Step 8: Commit**

```bash
git add database/client.py pipeline/fetch_functions.py
git commit -m "Add get_frontend_client(); convert fetch_functions.py off its module-level singleton client"
```

---

## Task 4: Update `app.py` to construct its client once in `server()`

**Files:**
- Modify: `app.py:7` (imports), `app.py:537-542` (`server()` setup), `app.py:778-779`, `app.py:958`, `app.py:979`, `app.py:1006`, `app.py:1056-1058`, `app.py:1110` (the six `fetch_*` call sites)

**Interfaces:**
- Consumes: `get_frontend_client` from `database.client` (Task 3), the seven `client`-first-parameter functions from `pipeline.fetch_functions` (Task 3).

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam")

import ast

src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam/app.py").read()
ast.parse(src)
assert "from database.client import get_frontend_client" in src
assert "client = get_frontend_client()" in src
# every fetch_* call site must now pass client as the first argument
for call in ["fetch_dm(client,", "fetch_rmse(client,", "fetch_nowcast_data(client,",
             "fetch_confidence_intervals(client,", "fetch_realised_gdp(client,",
             "fetch_historical_data(\n                    client,", "fetch_historical_data(client,"]:
    pass  # checked individually below for the two fetch_rmse/fetch_historical_data call-site variants

assert src.count("fetch_rmse(client,") == 2, f"expected 2 fetch_rmse call sites updated, found {src.count('fetch_rmse(client,')}"
assert "fetch_dm(client," in src
assert "fetch_nowcast_data(client," in src
assert "fetch_confidence_intervals(client," in src
assert "fetch_realised_gdp(client," in src
assert "fetch_historical_data(\n                    client," in src or "fetch_historical_data(client," in src

# app.py must NOT construct the client at module scope (only inside server())
module_level_src = src.split("def server(")[0]
assert "get_frontend_client()" not in module_level_src, "client must be built inside server(), not at module scope"

print("ALL TASK 4 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: from database.client import get_frontend_client not found`

- [ ] **Step 3: Update `app.py`'s imports**

Replace line 7:
```python
from pipeline.fetch_functions import fetch_nowcast_data, fetch_confidence_intervals, fetch_historical_data, fetch_rmse, fetch_dm, fetch_realised_gdp
```
with:
```python
from pipeline.fetch_functions import fetch_nowcast_data, fetch_confidence_intervals, fetch_historical_data, fetch_rmse, fetch_dm, fetch_realised_gdp
from database.client import get_frontend_client
```

- [ ] **Step 4: Construct the client once inside `server()`**

Replace (currently lines 537-542):
```python
def server(input, output, session):

    wizard_step = reactive.value(1)
    dm_overlay_visible = reactive.value(False)
    models_overlay_visible = reactive.value(False)
    is_dark = reactive.value(False)
```
with:
```python
def server(input, output, session):

    client = get_frontend_client()

    wizard_step = reactive.value(1)
    dm_overlay_visible = reactive.value(False)
    models_overlay_visible = reactive.value(False)
    is_dark = reactive.value(False)
```
Every `@render.*`/`@render_widget`-decorated function below this point is defined inside `server()`'s body, so `client` is visible to all of them via normal Python closure scoping — no Shiny reactive machinery needed for this.

- [ ] **Step 5: Update the six `fetch_*` call sites**

In the `dm_overlay()` function (currently around lines 778-779):
```python
        dm_pairs = fetch_dm(db_models, flash_month)
        metrics = fetch_rmse(db_models)
```
becomes:
```python
        dm_pairs = fetch_dm(client, db_models, flash_month)
        metrics = fetch_rmse(client, db_models)
```

In the `nowcast_plot()` function (currently around line 958):
```python
        data, x_labels = fetch_nowcast_data(quarter)
```
becomes:
```python
        data, x_labels = fetch_nowcast_data(client, quarter)
```

Still inside `nowcast_plot()` (currently around line 979):
```python
                x_ci, ci50_lo, ci50_hi, ci80_lo, ci80_hi = fetch_confidence_intervals(quarter, db_ci_model)
```
becomes:
```python
                x_ci, ci50_lo, ci50_hi, ci80_lo, ci80_hi = fetch_confidence_intervals(client, quarter, db_ci_model)
```

Still inside `nowcast_plot()` (currently around line 1006):
```python
        realised = fetch_realised_gdp(quarter)
```
becomes:
```python
        realised = fetch_realised_gdp(client, quarter)
```

In the historical-plot render function (currently around lines 1056-1058):
```python
        quarters, actual, predictions = fetch_historical_data(
                    start_date, end_date, flash_month
                )
```
becomes:
```python
        quarters, actual, predictions = fetch_historical_data(
                    client, start_date, end_date, flash_month
                )
```

In `eval_metrics()` (currently around line 1110):
```python
        metrics = fetch_rmse(db_models)
```
becomes:
```python
        metrics = fetch_rmse(client, db_models)
```

- [ ] **Step 6: Run the verification script again**

Expected: `ALL TASK 4 CHECKS PASSED`

- [ ] **Step 7: Confirm `app.py` imports cleanly without credentials (the actual bug this whole plan fixes)**

```python
import subprocess, os
stripped_env = {k: v for k, v in os.environ.items() if not k.startswith("SUPABASE")}
result = subprocess.run(
    ["python3", "-c", "import app; print('APP IMPORTED WITH NO SUPABASE ENV VARS')"],
    cwd="/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/supabase-client-seam",
    capture_output=True, text=True, env=stripped_env,
)
print(result.stdout[-1500:])
print(result.stderr[-1500:])
assert result.returncode == 0, f"app.py should import cleanly with no Supabase credentials at all: {result.stderr}"
assert "APP IMPORTED WITH NO SUPABASE ENV VARS" in result.stdout
print("APP IMPORT-WITHOUT-CREDENTIALS CHECK OK")
```

This confirms the concrete problem this whole plan set out to fix: `app.py` (and, transitively, its pure logic like `date_to_quarter`, `shift_quarter`, `to_db_names`, `from_db_name`) can now be imported and unit-tested without any live Supabase credentials present — something that was impossible before Task 3.

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "Build the Supabase client once in server(); pass it through to every fetch_functions call"
```

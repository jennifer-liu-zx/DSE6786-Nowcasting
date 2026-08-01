# Evaluation-Support Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse four duplicated pieces of evaluation logic (`fetch_all_model_forecasts`, `get_month_date`, the CI-band formula, `plot_poos_results`) into one new deep module, `pipeline/evaluation_support.py`, and delete every duplicate.

**Architecture:** One new module with zero dependencies on `poos.py`/`historical.py`/`prediction.py`, sitting below all of them in the import graph. Six existing files (`prediction.py`, `poos.py`, `ci_update.py`, `historical.py`, `evaluation_table_hist.py`, `plot_poos.py`) import from it instead of defining their own copies.

**Tech Stack:** Python 3.13, pandas.

## Global Constraints

- `compute_ci_bounds(point, rmse)` returns a plain tuple `(lb50, ub50, lb80, ub80)` — no dict, no forced key-naming convention. Callers assign the four values to whatever keys they already use.
- `plot_poos_results`'s canonical signature is `(y_full, y_df, model_name, version, last_n=200)` — the only signature any real caller uses. `poos.py`'s own copy (a different `title: str` signature) is dead code (defined, never called anywhere including within `poos.py` itself) — delete it outright, no replacement call needed in `poos.py`.
- `evaluation_table_hist.py`'s pasted-twice upsert block (lines 157-171, identical to 165-171) collapses to one upsert as a side effect of this consolidation — not a separate fix.
- No pytest/unittest exists in this repo — verification is standalone `python3` scripts with `assert` statements, matching the existing `if __name__ == "__main__":` idiom, same as the prior correctness-fix pass.
- Work happens in an isolated git worktree off current `main` (`1dd9cd0`). Every implementer must verify `pwd`/`git rev-parse HEAD`/`git branch --show-current` before doing anything, and again before committing — a prior pass had a subagent accidentally commit to the main checkout instead of the worktree.
- If any verification script uses `sys.path.insert(...)`, it must point at the worktree checkout, not the main repo path — a prior pass hit a bug where the wrong absolute path silently tested stale code.

---

## Task 1: Create `pipeline/evaluation_support.py`

**Files:**
- Create: `pipeline/evaluation_support.py`
- Test: standalone verification script (scratchpad)

**Interfaces:**
- Produces: `fetch_all_model_forecasts(client) -> pd.DataFrame`, `get_month_date(quarter_ts: pd.Timestamp, version: int) -> pd.Timestamp`, `compute_ci_bounds(point, rmse) -> tuple` (works for both scalar floats and pandas Series — plain arithmetic broadcasts over both), `plot_poos_results(y_full: pd.Series, y_df: pd.DataFrame, model_name: str, version: int, last_n: int = 200) -> None`.

- [ ] **Step 1: Write the failing verification script**

Create a scratchpad script (pick your own path under a scratch/temp directory) that imports from `pipeline.evaluation_support` and asserts:
```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support")  # the worktree checkout, not the main repo path

import pandas as pd
from pipeline.evaluation_support import fetch_all_model_forecasts, get_month_date, compute_ci_bounds, plot_poos_results

# get_month_date: version 1-6 maps to month offsets 0-5 from quarter start
q = pd.Timestamp("2024-01-01")
assert get_month_date(q, 1) == pd.Timestamp("2024-01-31")
assert get_month_date(q, 4) == pd.Timestamp("2024-04-30")
assert get_month_date(q, 6) == pd.Timestamp("2024-06-30")
try:
    get_month_date(q, 7)
    raise AssertionError("expected ValueError for out-of-range version")
except ValueError:
    pass

# compute_ci_bounds: works on scalars
lb50, ub50, lb80, ub80 = compute_ci_bounds(10.0, 2.0)
assert abs(lb50 - (10.0 - 0.674*2.0)) < 1e-9
assert abs(ub50 - (10.0 + 0.674*2.0)) < 1e-9
assert abs(lb80 - (10.0 - 1.282*2.0)) < 1e-9
assert abs(ub80 - (10.0 + 1.282*2.0)) < 1e-9

# compute_ci_bounds: works on a pandas Series (vectorized), matching ci_update.py's usage
s = pd.Series([10.0, 20.0])
r = pd.Series([2.0, 3.0])
lb50_s, ub50_s, lb80_s, ub80_s = compute_ci_bounds(s, r)
assert isinstance(lb50_s, pd.Series)
assert abs(lb50_s.iloc[1] - (20.0 - 0.674*3.0)) < 1e-9

# fetch_all_model_forecasts and plot_poos_results: import-only check (both need
# live data / a real client or matplotlib figure to fully exercise — confirm
# they're at least importable and have the right signature)
import inspect
sig = inspect.signature(fetch_all_model_forecasts)
assert list(sig.parameters) == ["client"]
sig2 = inspect.signature(plot_poos_results)
assert list(sig2.parameters) == ["y_full", "y_df", "model_name", "version", "last_n"]

print("ALL EVALUATION_SUPPORT CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `ModuleNotFoundError: No module named 'pipeline.evaluation_support'`

- [ ] **Step 3: Create `pipeline/evaluation_support.py`**

```python
"""
Shared evaluation-support code used across the prediction and POOS
(pseudo-out-of-sample) pipeline stages: paginated model_forecasts fetch,
version-to-month-date mapping, the Gaussian-quantile confidence-interval
formula, and POOS result plotting.

This module depends on nothing in pipeline.poos, pipeline.historical, or
pipeline.prediction — it sits below all three so any of them can import
from here without creating a circular import.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt


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
```

- [ ] **Step 4: Run the verification script again to confirm it passes**

Expected: `ALL EVALUATION_SUPPORT CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/evaluation_support.py
git commit -m "Add pipeline/evaluation_support.py: shared fetch/date/CI/plot helpers"
```

---

## Task 2: Migrate `pipeline/prediction.py` and `pipeline/poos.py`

**Files:**
- Modify: `pipeline/prediction.py`
- Modify: `pipeline/poos.py`

**Interfaces:**
- Consumes: `fetch_all_model_forecasts`, `compute_ci_bounds` from `pipeline.evaluation_support` (Task 1).

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support")

import ast

for f in ["pipeline/prediction.py", "pipeline/poos.py"]:
    src = open(f"/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support/{f}").read()
    ast.parse(src)
    assert "def fetch_all_model_forecasts(" not in src, f"{f} still defines its own fetch_all_model_forecasts"
    assert "def plot_poos_results(" not in src, f"{f} still defines its own plot_poos_results"
    assert "0.674" not in src and "1.282" not in src, f"{f} still has an inline CI formula"

print("ALL TASK 2 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: pipeline/prediction.py still defines its own fetch_all_model_forecasts`

- [ ] **Step 3: Edit `pipeline/prediction.py`**

Add to the imports near the top (after line 6, the last `from pipeline.models...` import):
```python
from pipeline.evaluation_support import fetch_all_model_forecasts, compute_ci_bounds
```

Delete the local `fetch_all_model_forecasts` definition (currently lines 9-28 — the whole function, including its two blank lines before `assign_version_prev`).

In `nowcast_single` (the block building the returned `pd.DataFrame`), replace:
```python
            "pred_50_lower": float(y_test_predicted) - 0.674 * rmse,
            "pred_50_upper": float(y_test_predicted) + 0.674 * rmse,
            "pred_80_lower": float(y_test_predicted) - 1.282 * rmse,
            "pred_80_upper": float(y_test_predicted) + 1.282 * rmse,
```
with:
```python
            "pred_50_lower": lb50,
            "pred_50_upper": ub50,
            "pred_80_lower": lb80,
            "pred_80_upper": ub80,
```
and add, immediately before that `return pd.DataFrame(...)` call:
```python
    lb50, ub50, lb80, ub80 = compute_ci_bounds(float(y_test_predicted), rmse)
```

Do the identical replacement in `nowcast_single_latest` (same four `"pred_*"` lines appear again further down the file — locate them by content, the line numbers will have shifted after the first edit).

Also fix the redundant client fetch: find the block (near the bottom of the file, in `prediction_pipeline`) that does:
```python
    supabase_client = get_backend_client()
    gdp_response = supabase_client.table("gdp").select("sasdate, GDPC1_t").execute()
    gdp_response = get_backend_client().table("gdp").select("sasdate, GDPC1_t").order("sasdate", desc=False).execute()
```
Replace with just:
```python
    supabase_client = get_backend_client()
    gdp_response = supabase_client.table("gdp").select("sasdate, GDPC1_t").order("sasdate", desc=False).execute()
```
(keeps the `.order(...)` call from the second, actually-used fetch; drops the first, wasted network round-trip whose result was discarded).

- [ ] **Step 4: Edit `pipeline/poos.py`**

Add near the top imports (after `import matplotlib.pyplot as plt`):
```python
from pipeline.evaluation_support import compute_ci_bounds
```

Delete the entire `plot_poos_results` function (currently lines 328-401, from the `# ── Plotting helper ──` comment through the function's closing `plt.close()`/end — it's dead code, never called anywhere including within this file).

In `poos_validation`, replace:
```python
                "pred_50_lower":  y_test_predicted - 0.674 * train_rmse,
                "pred_50_upper":  y_test_predicted + 0.674 * train_rmse,
                "pred_80_lower":  y_test_predicted - 1.282 * train_rmse,
                "pred_80_upper":  y_test_predicted + 1.282 * train_rmse,
```
with:
```python
                "pred_50_lower":  lb50,
                "pred_50_upper":  ub50,
                "pred_80_lower":  lb80,
                "pred_80_upper":  ub80,
```
and add, immediately before the `records.append({` line:
```python
        lb50, ub50, lb80, ub80 = compute_ci_bounds(y_test_predicted, train_rmse)
```

- [ ] **Step 5: Run the verification script again**

Expected: `ALL TASK 2 CHECKS PASSED`

- [ ] **Step 6: Confirm both files still parse and the CI values are numerically unchanged**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support")
import ast
for f in ["pipeline/prediction.py", "pipeline/poos.py"]:
    ast.parse(open(f"/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support/{f}").read())
    print(f, "OK")

from pipeline.evaluation_support import compute_ci_bounds
lb50, ub50, lb80, ub80 = compute_ci_bounds(5.0, 1.0)
assert abs(lb50 - (5.0 - 0.674)) < 1e-9
assert abs(ub80 - (5.0 + 1.282)) < 1e-9
print("NUMERIC CHECK OK")
```

- [ ] **Step 7: Commit**

```bash
git add pipeline/prediction.py pipeline/poos.py
git commit -m "Migrate prediction.py and poos.py onto pipeline.evaluation_support"
```

---

## Task 3: Migrate `pipeline/ci_update.py` and `pipeline/historical.py`

**Files:**
- Modify: `pipeline/ci_update.py`
- Modify: `pipeline/historical.py`

**Interfaces:**
- Consumes: `fetch_all_model_forecasts`, `get_month_date`, `compute_ci_bounds`, `plot_poos_results` from `pipeline.evaluation_support` (Task 1).

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support")

import ast

src_ci = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support/pipeline/ci_update.py").read()
ast.parse(src_ci)
assert "def fetch_all_model_forecasts(" not in src_ci
assert "def get_month_date(" not in src_ci
assert "z50 = 0.674" not in src_ci

src_hist = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support/pipeline/historical.py").read()
ast.parse(src_hist)
assert "def get_month_date(" not in src_hist
assert "def plot_poos_results(" not in src_hist

print("ALL TASK 3 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError` (both files still define their own copies).

- [ ] **Step 3: Edit `pipeline/ci_update.py`**

Change the imports at the top from:
```python
import pandas as pd
import numpy as np
from database.client import get_backend_client
```
to:
```python
import pandas as pd
import numpy as np
from database.client import get_backend_client
from pipeline.evaluation_support import fetch_all_model_forecasts, get_month_date, compute_ci_bounds
```

Delete the local `fetch_all_model_forecasts` definition (lines 6-25) and the local `get_month_date` definition (lines 28-35) — keep the blank lines around them tidy (no double-blank-line artifacts).

In `update_ci_columns`, replace:
```python
    # ── Compute CI ───────────────────────────────────────────────────────────
    z50 = 0.674
    z80 = 1.282

    merged["ci_50_lb"] = merged["nowcast"] - z50 * merged["rmse"]
    merged["ci_50_ub"] = merged["nowcast"] + z50 * merged["rmse"]
    merged["ci_80_lb"] = merged["nowcast"] - z80 * merged["rmse"]
    merged["ci_80_ub"] = merged["nowcast"] + z80 * merged["rmse"]
```
with:
```python
    # ── Compute CI ───────────────────────────────────────────────────────────
    merged["ci_50_lb"], merged["ci_50_ub"], merged["ci_80_lb"], merged["ci_80_ub"] = (
        compute_ci_bounds(merged["nowcast"], merged["rmse"])
    )
```

- [ ] **Step 4: Edit `pipeline/historical.py`**

Add to the imports near the top (after the existing `from pipeline.poos import poos_validation` line):
```python
from pipeline.evaluation_support import get_month_date, plot_poos_results
```

Delete the local `get_month_date` definition (currently lines 14-35, including its docstring) and the local `plot_poos_results` definition (currently lines 160-236, the whole function through its closing `plt.close()`).

Everything else in `historical.py` (`push_poos_to_supabase`, `push_evaluation_to_supabase`, `BUILD_REGISTRY`, `MODEL_REGISTRY`, `run()`) stays as-is — they already call `get_month_date`/`plot_poos_results` by name, which now resolves to the imported versions.

- [ ] **Step 5: Run the verification script again**

Expected: `ALL TASK 3 CHECKS PASSED`

- [ ] **Step 6: Confirm both files still parse and get_month_date's behavior is unchanged**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support")
import ast
for f in ["pipeline/ci_update.py", "pipeline/historical.py"]:
    ast.parse(open(f"/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support/{f}").read())
    print(f, "OK")

import pandas as pd
from pipeline.historical import get_month_date  # re-exported via import, should still resolve
assert get_month_date(pd.Timestamp("2024-01-01"), 3) == pd.Timestamp("2024-03-31")
print("BEHAVIOR CHECK OK")
```

- [ ] **Step 7: Commit**

```bash
git add pipeline/ci_update.py pipeline/historical.py
git commit -m "Migrate ci_update.py and historical.py onto pipeline.evaluation_support"
```

---

## Task 4: Migrate `pipeline/evaluation_table_hist.py` and `pipeline/plot_poos.py`

**Files:**
- Modify: `pipeline/evaluation_table_hist.py`
- Modify: `pipeline/plot_poos.py`

**Interfaces:**
- Consumes: `fetch_all_model_forecasts` from `pipeline.evaluation_support` (both files); `get_month_date`, `plot_poos_results` from `pipeline.evaluation_support` (`plot_poos.py` only, replacing its current imports from `ci_update`/`historical`).

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support")

import ast

src_eth = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support/pipeline/evaluation_table_hist.py").read()
ast.parse(src_eth)
assert "def fetch_all_model_forecasts(" not in src_eth
assert src_eth.count('client.table("evaluation").upsert(') == 1, "double-upsert not fixed"
assert src_eth.count("import pandas as pd") == 1, "duplicate top-level imports not cleaned up"

src_pp = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support/pipeline/plot_poos.py").read()
ast.parse(src_pp)
assert "from pipeline.ci_update import" not in src_pp
assert "from pipeline.historical import" not in src_pp
assert "from pipeline.evaluation_support import" in src_pp

print("ALL TASK 4 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError` (multiple — double-upsert still present, duplicate imports still present, `plot_poos.py` still imports from the old locations).

- [ ] **Step 3: Edit `pipeline/evaluation_table_hist.py`**

Replace the entire top of the file (currently lines 1-29, which duplicate `import pandas as pd`/`from database.client import get_backend_client` twice with a stray `from supabase import client` in between) with a single clean import block:
```python
import pandas as pd
import numpy as np
from database.client import get_backend_client
from pipeline.evaluation_support import fetch_all_model_forecasts
```

Delete the local `fetch_all_model_forecasts` definition entirely (it's part of what you just replaced above).

Delete the second block of duplicate imports before `calculate_and_upsert_rmse` (currently lines 174-181: `import pandas as pd` / `from database.client import get_backend_client` / `import numpy as np`, repeated twice) — `numpy` is already imported once at the top per the replacement above, and `pandas`/`get_backend_client` likewise; nothing in this file needs re-importing mid-file.

Fix the double-upsert in `push_forecasts_to_evaluation`: delete the second, redundant block (currently lines 165-171 — the exact repeat of the block at lines 157-163), leaving exactly one:
```python
    # ── Upsert ───────────────────────────────────────────────────────────────
    client.table("evaluation").upsert(
        records,
        on_conflict="quarter_date,version"
    ).execute()

    print(f"Upserted {len(records)} rows into 'evaluation'.")
```

- [ ] **Step 4: Edit `pipeline/plot_poos.py`**

Change:
```python
from pipeline.ci_update import fetch_all_model_forecasts, get_month_date
from pipeline.historical import plot_poos_results
```
to:
```python
from pipeline.evaluation_support import fetch_all_model_forecasts, get_month_date, plot_poos_results
```

Nothing else in this file changes — `infer_version`, `run()`, and every call site already use these three names as imported.

- [ ] **Step 5: Run the verification script again**

Expected: `ALL TASK 4 CHECKS PASSED`

- [ ] **Step 6: Full-repo consistency sweep**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support")
import ast

files = [
    "pipeline/evaluation_support.py", "pipeline/prediction.py", "pipeline/poos.py",
    "pipeline/ci_update.py", "pipeline/historical.py",
    "pipeline/evaluation_table_hist.py", "pipeline/plot_poos.py",
]
for f in files:
    src = open(f"/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support/{f}").read()
    ast.parse(src)
    print(f, "OK")

# No file outside evaluation_support.py should define these anymore
import subprocess
for fn in ["def fetch_all_model_forecasts(", "def get_month_date(", "def plot_poos_results("]:
    out = subprocess.run(
        ["grep", "-rl", fn, "pipeline/"],
        cwd="/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support", capture_output=True, text=True
    ).stdout.strip().splitlines()
    assert out == ["pipeline/evaluation_support.py"], f"{fn!r} found outside evaluation_support.py: {out}"

print("FULL REPO SWEEP OK")
```

Also run the baseline import smoke test used throughout the prior correctness-fix pass, extended to cover every touched module:
```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/evaluation-support")
import pipeline.evaluation_support, pipeline.prediction, pipeline.poos, pipeline.ci_update, pipeline.historical, pipeline.evaluation_table_hist, pipeline.plot_poos
print("IMPORTS OK")
```

- [ ] **Step 7: Commit**

```bash
git add pipeline/evaluation_table_hist.py pipeline/plot_poos.py
git commit -m "Migrate evaluation_table_hist.py and plot_poos.py onto pipeline.evaluation_support; fix double-upsert"
```

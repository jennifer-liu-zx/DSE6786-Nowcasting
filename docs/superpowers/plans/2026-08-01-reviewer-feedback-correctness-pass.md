# Reviewer Feedback Correctness Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confirmed bugs and misleading descriptions across the nowcasting pipeline, database schema, frontend, and Technical Documentation.docx, per the design spec at `docs/superpowers/specs/2026-08-01-reviewer-feedback-correctness-pass-design.md`.

**Architecture:** No new subsystems. Each task is a targeted, self-contained fix to an existing file, verified with a standalone `python3` verification script (this codebase has no pytest/unittest infrastructure — its established pattern is `if __name__ == "__main__":` smoke blocks and standalone scripts, so verification here follows that same idiom instead of introducing a new test framework unilaterally).

**Tech Stack:** Python 3.13, pandas, statsmodels, python-docx, Shiny for Python, Supabase.

## Global Constraints

- No live Supabase connection is available or required for any task in this plan (project is paused; restoring it is explicitly out of scope). Verification uses local CSV fixtures at `/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data/` (already present on this machine) instead.
- No pytest/unittest exists in this repo — do not introduce it. Verification scripts are plain Python with `assert` statements, matching the existing `if __name__ == "__main__":` idiom already used in `pipeline/ragged_edge.py`, `pipeline/poos.py`, `pipeline/output_x.py`.
- Do not fabricate or hand-edit any RMSE/DM numeric values in the doc — the full historical POOS re-run that would produce correct numbers is out of scope for this pass (user will run it later). Where a doc table's numbers are stale after a code change, add an explicit note instead of editing the numbers.
- AR(2) lag order everywhere (live nowcast path and POOS evaluation) — confirmed target spec.
- DM test loss function: `"squared"`, to match RMSE reporting.
- GDP components PCECC96, GPDIC1, EXPGSC1: lag-only (never contemporaneous).
- A venv with pandas/statsmodels/scipy/python-docx already exists at `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python` — reuse it for verification scripts rather than installing into the project's own environment.

---

## Task 1: AR benchmark — direct-forecast rewrite + AR(2)/AR(4) leak fix

**Files:**
- Modify: `pipeline/output_x.py:308-325` (`build_X_AR`)
- Modify: `pipeline/output_x_poos.py:150-167` (`build_X_AR_from_cut`), `pipeline/output_x_poos.py:212-213` (`make_build_X`'s `"X_AR"` case)
- Modify: `pipeline/poos.py:55-142` (`cut_and_fill`), `pipeline/poos.py:147-235` (`poos_validation`)
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_ar_benchmark.py`

**Interfaces:**
- Consumes: `pipeline.models.AR_benchmark.ar_model_nowcast(X, y)` — unchanged, still does `X_train = X.iloc[:-1]`, `X_test = X.iloc[[-1]]`, OLS with intercept. This task does not touch that function.
- Produces: `build_X_AR(n_lags: int = 2) -> tuple[pd.DataFrame, pd.Series]` returning exactly 2 feature columns named `lag_a`, `lag_b` (generic names regardless of which real lags they represent). `build_X_AR_from_cut(gdp_cut: pd.Series, gdp_actual: pd.Series, n_lags: int = 2) -> tuple[pd.DataFrame, pd.Series]` with the same 2-column `lag_a`/`lag_b` contract. `cut_and_fill(...)` now returns a 4-tuple `(qd_filled, md_filled, gdp_filled, gdp_raw)` instead of a 3-tuple.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_ar_benchmark.py`:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")

import pandas as pd
from pipeline.output_x_poos import build_X_AR_from_cut, make_build_X

DATA_DIR = "/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data"

gdp_df = pd.read_csv(f"{DATA_DIR}/gdp.csv")
gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
gdp = gdp_df.set_index("sasdate")["GDPC1_t"].astype(float)

# Case 1: t-1 IS available -> should use standard 1-step AR(2) (lag_1, lag_2)
gdp_cut_full = gdp[gdp.index <= pd.Timestamp("2024-12-01")]
X_full, y_full = build_X_AR_from_cut(gdp_cut_full, gdp)
assert list(X_full.columns) == ["lag_a", "lag_b"], f"expected [lag_a, lag_b], got {list(X_full.columns)}"
# lag_a for the last (target) row should equal the true t-1 value
target_date = gdp_cut_full.index[-1] + pd.DateOffset(months=3)
assert abs(X_full.loc[target_date, "lag_a"] - gdp_cut_full.iloc[-1]) < 1e-9, \
    "expected lag_a to be the real t-1 value when t-1 is available"

# Case 2: t-1 is MISSING (simulate un-released quarter) -> direct 2-step (lag_2, lag_3)
gdp_cut_gap = gdp_cut_full.iloc[:-1]  # drop the most recent quarter -> t-1 unavailable
X_gap, y_gap = build_X_AR_from_cut(gdp_cut_gap, gdp)
target_date_gap = gdp_cut_gap.index[-1] + pd.DateOffset(months=3) + pd.DateOffset(months=3)
assert abs(X_gap.loc[target_date_gap, "lag_a"] - gdp_cut_gap.iloc[-1]) < 1e-9, \
    "expected lag_a to fall back to the real t-2 value when t-1 is unavailable"

# Case 3: leak fix — make_build_X's outer n_lags (default 4, meant for X1-X4)
# must NOT change the AR builder's own 2-lag behaviour.
build_fn_default = make_build_X("X_AR")               # outer n_lags defaults to 4
build_fn_explicit4 = make_build_X("X_AR", n_lags=4)    # explicitly 4, historically leaked through
X_default, _ = build_fn_default(qd=None, md=None, gdp_cut=gdp_cut_full, gdp_actual=gdp)
X_leak, _ = build_fn_explicit4(qd=None, md=None, gdp_cut=gdp_cut_full, gdp_actual=gdp)
assert list(X_default.columns) == ["lag_a", "lag_b"], "X_AR must always be 2 columns regardless of outer n_lags"
assert list(X_leak.columns) == ["lag_a", "lag_b"], "X_AR must always be 2 columns regardless of outer n_lags"

print("ALL AR BENCHMARK CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_ar_benchmark.py`
Expected: `AssertionError` (columns are currently `lag_1`, `lag_2`, not `lag_a`, `lag_b`, and there's no direct-forecast fallback).

- [ ] **Step 3: Rewrite `build_X_AR_from_cut` in `pipeline/output_x_poos.py`**

Replace the existing function (lines 150-167) with:

```python
def build_X_AR_from_cut(
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_lags: int = 2,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Minimal GDP-only AR benchmark with a direct multi-step fallback.

    gdp_cut must be the RAW (un-imputed) GDP series cut at the information
    set available at prediction time — no ensemble/mean fill-in. If the
    quarter immediately before the target (t-1) is released, uses a
    standard 1-step AR(2): lag_a = t-1, lag_b = t-2. If t-1 is NOT yet
    released, uses a direct 2-step forecast instead of imputing the
    missing lag: lag_a = t-2, lag_b = t-3.
    """
    target_date = gdp_cut.index[-1] + relativedelta(months=3)
    full_index  = gdp_cut.index.append(pd.DatetimeIndex([target_date]))
    gdp_reindexed = gdp_cut.reindex(full_index)

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
```

(`relativedelta` is already imported at `pipeline/output_x_poos.py:169` — move that import to the top of the file alongside the existing `import pandas as pd`.)

- [ ] **Step 4: Fix the `make_build_X` leak in `pipeline/output_x_poos.py:212-213`**

Change:
```python
        case "X_AR":
            return lambda qd, md, gdp_cut, gdp_actual: build_X_AR_from_cut(gdp_cut, gdp_actual, n_lags)
```
to:
```python
        case "X_AR":
            return lambda qd, md, gdp_cut, gdp_actual: build_X_AR_from_cut(gdp_cut, gdp_actual)
```
(drop the `n_lags` pass-through entirely so `build_X_AR_from_cut` always uses its own default of 2, regardless of the outer `n_lags` used for X1–X4.)

- [ ] **Step 5: Rewrite `build_X_AR` in `pipeline/output_x.py:308-325`**

Replace with:

```python
def build_X_AR(n_lags: int = 2) -> tuple:
    """
    X_AR: minimal GDP-only AR benchmark, direct-forecast on real lags only.

    Unlike the other builders, this does NOT use _load_gdp_with_flash() —
    the benchmark must not depend on the ensemble's own predictions. If
    GDP at t-1 is released, uses a standard 1-step AR(2) (lag_a=t-1,
    lag_b=t-2). If t-1 is not yet released, uses a direct 2-step forecast
    (lag_a=t-2, lag_b=t-3) instead of imputing the missing lag.
    """
    y_gdp = _load_gdp()["GDPC1_t"]
    df = y_gdp.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 2):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)

    t1_available = pd.notna(df["lag_1"].iloc[-1])
    lag_cols = ["lag_1", "lag_2"] if t1_available else ["lag_2", "lag_3"]

    df_hist = df.iloc[:-1][["gdp_growth"] + lag_cols].dropna()
    X_hist = df_hist[lag_cols].rename(columns={lag_cols[0]: "lag_a", lag_cols[1]: "lag_b"})
    y_hist = df_hist["gdp_growth"]

    last_row = df.iloc[[-1]]
    X_last = last_row[lag_cols].rename(columns={lag_cols[0]: "lag_a", lag_cols[1]: "lag_b"})
    y_last = last_row["gdp_growth"]

    X = pd.concat([X_hist, X_last])
    y = pd.concat([y_hist, y_last])
    print(f"X_AR (direct-forecast, using {lag_cols}): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y
```

- [ ] **Step 6: Give AR the raw (un-imputed) GDP series in `pipeline/poos.py`**

In `cut_and_fill` (lines 55-142), immediately after the raw cut (currently line 90: `gdp_cut = gdp[gdp.index <= pd.Timestamp(gdp_cutoff)].copy()`), add:
```python
    gdp_cut_raw = gdp_cut.copy()
```
Then change the function's final `return` statement (currently line 142: `return qd_filled, md_filled, gdp_cut`) to:
```python
    return qd_filled, md_filled, gdp_cut, gdp_cut_raw
```

In `poos_validation` (lines 147-235), update the call site (currently lines 179-185):
```python
        qd_filled, md_filled, gdp_cutoff = cut_and_fill(
            version=version,
            q_predicted=pd.Timestamp(q_predicted),
            QD_t=QD_t,
            MD_t=MD_t,
            gdp=y_full
        )
```
to:
```python
        qd_filled, md_filled, gdp_filled, gdp_raw = cut_and_fill(
            version=version,
            q_predicted=pd.Timestamp(q_predicted),
            QD_t=QD_t,
            MD_t=MD_t,
            gdp=y_full
        )
        gdp_for_build = gdp_raw if buildname == "X_AR" else gdp_filled
```
Then update the `buildfn(...)` call (currently lines 190-195) to use `gdp_for_build` instead of `gdp_cutoff`:
```python
        X, y = buildfn(
            qd=qd_filled,
            md=md_filled,
            gdp_cut=gdp_for_build,
            gdp_actual=y_full,
        )
```

- [ ] **Step 7: Fix `poos.py`'s own `__main__` smoke test, which also unpacks `cut_and_fill`'s return value**

`pipeline/poos.py:394-422` has an `if __name__ == "__main__":` smoke-test block that calls `cut_and_fill` and unpacks a 3-tuple — this breaks now that Step 6 changed it to a 4-tuple. In `pipeline/poos.py`, change (lines 406-413):
```python
    # Smoke-test cut_and_fill
    filled_qd, filled_md, gdp_filled = cut_and_fill(
        version=4,
        q_predicted=pd.Timestamp("2025-12-01"),
        QD_t=qd,
        MD_t=md,
        gdp=gdp_df["GDPC1_t"]
    )
    print("QD tail:"); print(filled_qd.tail())
    print("MD tail:"); print(filled_md.tail())
    print("GDP tail:"); print(gdp_filled.tail())
```
to:
```python
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
```

- [ ] **Step 8: Run the verification script again to confirm it passes**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_ar_benchmark.py`
Expected: `ALL AR BENCHMARK CHECKS PASSED`

- [ ] **Step 9: Commit**

```bash
git add pipeline/output_x.py pipeline/output_x_poos.py pipeline/poos.py
git commit -m "Fix AR benchmark: direct-forecast on real lags, decouple from X1-X4 n_lags"
```

---

## Task 2: DM test — squared-loss default

**Files:**
- Modify: `pipeline/dm_test.py:100`
- Test: inline (see Step 1)

**Interfaces:**
- Consumes: nothing new.
- Produces: `dm_test(y_actual, y_hat1, y_hat2, loss="squared", h=1, power=2.0, bandwidth="auto")` — same signature, only the default value of `loss` changes.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_dm_loss.py`:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")
import inspect
from pipeline.dm_test import dm_test

sig = inspect.signature(dm_test)
default_loss = sig.parameters["loss"].default
assert default_loss == "squared", f"expected default loss='squared', got '{default_loss}'"
print("DM LOSS DEFAULT CHECK PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_dm_loss.py`
Expected: `AssertionError: expected default loss='squared', got 'absolute'`

- [ ] **Step 3: Fix the default**

In `pipeline/dm_test.py:100`, change:
```python
    loss: str = "absolute",
```
to:
```python
    loss: str = "squared",
```

- [ ] **Step 4: Run the verification script again**

Run: same command as Step 2.
Expected: `DM LOSS DEFAULT CHECK PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/dm_test.py
git commit -m "Fix DM test to default to squared loss, matching reported RMSE"
```

---

## Task 3: GDP components — lag-only treatment

**Files:**
- Modify: `pipeline/output_x.py` (add module-level constant + drop calls in `build_X1`, `build_X2`, `build_X3`, `build_X4`)
- Modify: `pipeline/output_x_poos.py` (same, for the `_from_cut` builders)
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_gdp_components.py`

**Interfaces:**
- Produces: `GDP_COMPONENT_COLS = ["PCECC96_t_qd", "GPDIC1_t_qd", "EXPGSC1_t_qd"]` defined at module level in both `pipeline/output_x.py` and `pipeline/output_x_poos.py`. `build_X1`/`build_X3` (and their POOS twins) never contain these column names. `build_X2`/`build_X4` (and their POOS twins) never contain the un-suffixed names but do contain `{col}_lag1`...`{col}_lagN`.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_gdp_components.py`:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")

import pandas as pd
from pipeline.output_x import build_X1, build_X2, build_X3, build_X4, GDP_COMPONENT_COLS

DATA_DIR = "/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data"
df_md = pd.read_csv(f"{DATA_DIR}/filled_md.csv")
df_qd = pd.read_csv(f"{DATA_DIR}/filled_qd.csv")

assert GDP_COMPONENT_COLS == ["PCECC96_t_qd", "GPDIC1_t_qd", "EXPGSC1_t_qd"]

X1, _ = build_X1(df_md, df_qd)
X3, _ = build_X3(df_md, df_qd)
for col in GDP_COMPONENT_COLS:
    assert col not in X1.columns, f"{col} must not appear in X1 (no-lag design matrix)"
    assert col not in X3.columns, f"{col} must not appear in X3 (no-lag design matrix)"

X2, _ = build_X2(df_md, df_qd, n_lags=4)
X4, _ = build_X4(df_md, df_qd, n_monthly_lags=4, n_qd_lags=4)
for col in GDP_COMPONENT_COLS:
    assert col not in X2.columns, f"{col} (contemporaneous) must not appear in X2"
    assert f"{col}_lag1" in X2.columns, f"{col}_lag1 must still appear in X2"
    assert col not in X4.columns, f"{col} (contemporaneous) must not appear in X4"
    assert f"{col}_lag1" in X4.columns, f"{col}_lag1 must still appear in X4"

print("ALL GDP COMPONENT CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_gdp_components.py`
Expected: `ImportError: cannot import name 'GDP_COMPONENT_COLS'` (doesn't exist yet).

- [ ] **Step 3: Add the constant and fix the four builders in `pipeline/output_x.py`**

Add near the top of the file, after the `PROJECT_DIR` block (after line 40):
```python
# Real GDP components (consumption, investment, exports) that would let a
# model near-reconstruct GDP if used contemporaneously. Kept for their
# predictive lag information, but never as same-quarter regressors.
GDP_COMPONENT_COLS = ["PCECC96_t_qd", "GPDIC1_t_qd", "EXPGSC1_t_qd"]
```

In `build_X1` (around line 229), change:
```python
    X = df_avg.join(df_q, how="inner").join(_load_gdp_lags(), how="left")
```
to:
```python
    df_q = df_q.drop(columns=GDP_COMPONENT_COLS, errors="ignore")
    X = df_avg.join(df_q, how="inner").join(_load_gdp_lags(), how="left")
```

In `build_X2` (around line 250), change:
```python
    X = _add_lags(qd1, n_lags).join(_load_gdp_lags(), how="left")
```
to:
```python
    X = _add_lags(qd1, n_lags).drop(columns=GDP_COMPONENT_COLS, errors="ignore").join(_load_gdp_lags(), how="left")
```

In `build_X3` (around line 270), change:
```python
    X = df_umidas.join(df_q, how="inner").join(_load_gdp_lags(), how="left")
```
to:
```python
    df_q = df_q.drop(columns=GDP_COMPONENT_COLS, errors="ignore")
    X = df_umidas.join(df_q, how="inner").join(_load_gdp_lags(), how="left")
```

In `build_X4` (around line 296), change:
```python
    df_q_lagged      = _add_lags(df_q, n_qd_lags)
```
to:
```python
    df_q_lagged      = _add_lags(df_q, n_qd_lags).drop(columns=GDP_COMPONENT_COLS, errors="ignore")
```

- [ ] **Step 4: Mirror the same fix in `pipeline/output_x_poos.py`**

Add near the top of the file (after the `import pandas as pd` line):
```python
GDP_COMPONENT_COLS = ["PCECC96_t_qd", "GPDIC1_t_qd", "EXPGSC1_t_qd"]
```

In `build_X1_from_cut` (around line 72-73), change:
```python
    df_q       = _prep_qd_from_df(qd_filled)
    X_base     = df_avg.join(df_q, how="inner")
```
to:
```python
    df_q       = _prep_qd_from_df(qd_filled).drop(columns=GDP_COMPONENT_COLS, errors="ignore")
    X_base     = df_avg.join(df_q, how="inner")
```

In `build_X2_from_cut` (around line 89-90), change:
```python
    df_q       = _prep_qd_from_df(qd_filled)
    X_base     = _add_lags_df(df_avg.join(df_q, how="inner"), n_lags)
```
to:
```python
    df_q       = _prep_qd_from_df(qd_filled)
    X_base     = _add_lags_df(df_avg.join(df_q, how="inner"), n_lags).drop(columns=GDP_COMPONENT_COLS, errors="ignore")
```

In `build_X3_from_cut` (around line 107-108), change:
```python
    df_q       = _prep_qd_from_df(qd_filled)
    X_base     = df_umidas.join(df_q, how="inner")
```
to:
```python
    df_q       = _prep_qd_from_df(qd_filled).drop(columns=GDP_COMPONENT_COLS, errors="ignore")
    X_base     = df_umidas.join(df_q, how="inner")
```

In `build_X4_from_cut` (around line 129), change:
```python
    df_q_lagged      = _add_lags_df(df_q, n_qd_lags)
```
to:
```python
    df_q_lagged      = _add_lags_df(df_q, n_qd_lags).drop(columns=GDP_COMPONENT_COLS, errors="ignore")
```

- [ ] **Step 5: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL GDP COMPONENT CHECKS PASSED`

- [ ] **Step 6: Commit**

```bash
git add pipeline/output_x.py pipeline/output_x_poos.py
git commit -m "Treat GDP components (PCECC96, GPDIC1, EXPGSC1) as lag-only regressors"
```

---

## Task 4: Ragged-edge — harden `fillna(0)` into an explicit invariant check

**Files:**
- Modify: `pipeline/ragged_edge.py:109-171` (`fill_ragged_edge`, `fill_ragged_edge_until`)
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_ragged_edge.py`

**Interfaces:**
- Produces: `fill_ragged_edge(...)` and `fill_ragged_edge_until(...)` keep their existing return signatures, but now raise `ValueError` instead of silently zero-filling if any variable in `bic_lags.csv` still has NaNs after the AR-fill step.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_ragged_edge.py`:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")

import pandas as pd
from pipeline.ragged_edge import fill_ragged_edge_until

DATA_DIR = "/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data"
qd = pd.read_csv(f"{DATA_DIR}/fred_qd_X.csv")
md = pd.read_csv(f"{DATA_DIR}/fred_md.csv")

# Case 1: real data has no leftover NaNs (verified separately) -> should run clean, no error
filled_qd, filled_md = fill_ragged_edge_until(qd, md, cutoff_date="2026-06-01")
lag_df = pd.read_csv(f"{DATA_DIR}/bic_lags.csv")
tracked_cols = [c for c in lag_df["variable"] if c in filled_qd.columns or c in filled_md.columns]
assert len(tracked_cols) > 0
for col in tracked_cols:
    series = filled_qd[col] if col in filled_qd.columns else filled_md[col]
    assert series.notna().all(), f"{col} still has NaN after fill on real data (unexpected)"

# Case 2: inject an unfillable leading gap (variable with NO data at all) -> must raise, not silently zero
qd_broken = qd.copy()
qd_broken["PCECC96_t_qd"] = pd.NA
try:
    fill_ragged_edge_until(qd_broken, md, cutoff_date="2026-06-01")
    raise AssertionError("expected fill_ragged_edge_until to raise on an unfillable column, but it didn't")
except ValueError as e:
    assert "PCECC96_t_qd" in str(e), f"error message should name the offending column, got: {e}"

print("ALL RAGGED EDGE CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_ragged_edge.py`
Expected: `AssertionError: expected fill_ragged_edge_until to raise...` (current code silently zero-fills instead).

- [ ] **Step 3: Add a shared invariant-check helper and use it in both functions**

Add a new helper function above `fill_ragged_edge` in `pipeline/ragged_edge.py`:
```python
def _assert_no_leftover_nan(df: pd.DataFrame, lag_dict: dict, label: str) -> None:
    """
    _fill_series should resolve every tracked variable (interpolate + AR
    forecast). Any leftover NaN means a variable has no usable history at
    all within this window — silently zero-filling it would corrupt model
    training, so we fail loudly instead.
    """
    tracked_cols = [c for c in lag_dict if c in df.columns]
    nan_counts = df[tracked_cols].isna().sum()
    offending = nan_counts[nan_counts > 0]
    if not offending.empty:
        raise ValueError(
            f"{label}: {len(offending)} variable(s) still have NaN after "
            f"ragged-edge fill (would previously have been silently zeroed): "
            f"{offending.to_dict()}"
        )
```

In `fill_ragged_edge` (around line 109-140), replace:
```python
    df_filled = df_filled.fillna(0)
    print(f"Done. Final shape: {df_filled.shape}")

    return df_filled
```
with:
```python
    _assert_no_leftover_nan(df_filled, lag_dict, label=f"fill_ragged_edge({data_table})")
    print(f"Done. Final shape: {df_filled.shape}")

    return df_filled
```

In `fill_ragged_edge_until` (around line 142-171), replace:
```python
    QD_filled = QD_filled.fillna(0)
    MD_filled = MD_filled.fillna(0)

    return QD_filled, MD_filled
```
with:
```python
    _assert_no_leftover_nan(QD_filled, lag_dict, label="fill_ragged_edge_until(QD)")
    _assert_no_leftover_nan(MD_filled, lag_dict, label="fill_ragged_edge_until(MD)")

    return QD_filled, MD_filled
```

- [ ] **Step 4: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL RAGGED EDGE CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/ragged_edge.py
git commit -m "Replace silent fillna(0) with an explicit no-leftover-NaN invariant check"
```

---

## Task 5: `generate_schema.py` — fix `dm_test`, add `evaluation` and `rmse` tables

**Files:**
- Modify: `generate_schema.py`
- Regenerate: `database/schema.sql`
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_schema.py`

**Interfaces:**
- Produces: `database/schema.sql` containing valid `CREATE TABLE` statements for `dm_test` (TEXT model columns, `PRIMARY KEY (version, model_1, model_2)`), `evaluation` (`PRIMARY KEY (quarter_date, version)`), and `rmse` (`PRIMARY KEY (model, version)`) — matching what `pipeline/dm_test.py::push_dm_results_to_supabase`, `pipeline/evaluation_table_hist.py::push_forecasts_to_evaluation`, and `pipeline/evaluation_table_hist.py::calculate_and_upsert_rmse`/`calculate_mean_rmse_by_model` actually write.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_schema.py`:

```python
schema = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/database/schema.sql").read()

assert 'CREATE TABLE IF NOT EXISTS dm_test' in schema
dm_test_block = schema.split("CREATE TABLE IF NOT EXISTS dm_test")[1].split(");")[0]
assert '"model_1"         TEXT' in dm_test_block or "model_1         TEXT" in dm_test_block, \
    "dm_test.model_1 must be TEXT, not NUMERIC"
assert "PRIMARY KEY (version, model_1, model_2)" in dm_test_block, \
    "dm_test PK must be (version, model_1, model_2), matching the upsert conflict key"
assert "PRIMARY KEY (sasdate)" not in dm_test_block, \
    "dm_test must not reference a nonexistent sasdate column"

assert 'CREATE TABLE IF NOT EXISTS evaluation' in schema
assert 'CREATE TABLE IF NOT EXISTS rmse' in schema
eval_block = schema.split("CREATE TABLE IF NOT EXISTS evaluation")[1].split(");")[0]
assert "PRIMARY KEY (quarter_date, version)" in eval_block
rmse_block = schema.split("CREATE TABLE IF NOT EXISTS rmse")[1].split(");")[0]
assert "PRIMARY KEY (model, version)" in rmse_block

print("ALL SCHEMA CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_schema.py`
Expected: `AssertionError: dm_test.model_1 must be TEXT, not NUMERIC` (or an earlier assertion, since `evaluation`/`rmse` don't exist in schema.sql at all yet).

- [ ] **Step 3: Add the three table generators to `generate_schema.py`**

Add these three functions after `generate_model_forecasts_table()` (after line 125):

```python
def generate_dm_test_table() -> str:
    return "\n".join([
        f"-- {'─' * 60}",
        f"-- TABLE: dm_test",
        f"-- {'─' * 60}",
        f"",
        f"CREATE TABLE IF NOT EXISTS dm_test (",
        f"    version         NUMERIC     NOT NULL,",
        f"    model_1         TEXT        NOT NULL,",
        f"    model_2         TEXT        NOT NULL,",
        f"    test_statistic  NUMERIC,",
        f"    p_value         NUMERIC,",
        f"    PRIMARY KEY (version, model_1, model_2)",
        f");",
        f"",
    ])


def generate_evaluation_table() -> str:
    model_cols = [
        "AR_Benchmark", "RF_Lags_Average", "RF_Lags_UMIDAS",
        "LASSO_UMIDAS", "LASSO_Average", "LASSO_Lags_Average",
        "All_Model_Average",
    ]
    col_lines = "\n".join(f'    "{c}" NUMERIC,' for c in model_cols)
    return "\n".join([
        f"-- {'─' * 60}",
        f"-- TABLE: evaluation",
        f"-- {'─' * 60}",
        f"",
        f"CREATE TABLE IF NOT EXISTS evaluation (",
        f"    quarter_date    DATE        NOT NULL,",
        f"    version         NUMERIC     NOT NULL,",
        f"    month_date      DATE,",
        f"    gdp_actual      NUMERIC,",
        col_lines,
        f"    PRIMARY KEY (quarter_date, version)",
        f");",
        f"",
    ])


def generate_rmse_table() -> str:
    return "\n".join([
        f"-- {'─' * 60}",
        f"-- TABLE: rmse",
        f"-- {'─' * 60}",
        f"",
        f"CREATE TABLE IF NOT EXISTS rmse (",
        f"    model    TEXT     NOT NULL,",
        f"    version  NUMERIC  NOT NULL,",
        f"    rmse     NUMERIC,",
        f"    PRIMARY KEY (model, version)",
        f");",
        f"",
    ])
```

- [ ] **Step 4: Remove the stale `dm_test` CSV mapping and wire the new generators into `main()`**

Change `CSV_FILES` (lines 18-25) — remove the `"dm_test"` entry:
```python
CSV_FILES = {
    "gdp": DATA_DIR / "gdp.csv",
    "fred_md": DATA_DIR / "fred_md.csv",
    "fred_qd_x": DATA_DIR / "fred_qd_X.csv",
    "filled_md": DATA_DIR / "filled_md.csv",
    "filled_qd": DATA_DIR / "filled_qd.csv",
}
```

In `main()` (around line 150-151), change:
```python
    blocks.append(generate_model_forecasts_table())
    blocks.append(gen_rls("model_forecasts"))
```
to:
```python
    blocks.append(generate_model_forecasts_table())
    blocks.append(gen_rls("model_forecasts"))
    blocks.append(generate_dm_test_table())
    blocks.append(gen_rls("dm_test"))
    blocks.append(generate_evaluation_table())
    blocks.append(gen_rls("evaluation"))
    blocks.append(generate_rmse_table())
    blocks.append(gen_rls("rmse"))
```

- [ ] **Step 5: Regenerate `database/schema.sql`**

The other 5 tables are still CSV-header-driven and need real CSV files present. Run from the project root, pointing `DATA_DIR` at the local machine's existing data folder (these files are gitignored in this repo but exist there):

```bash
cd /Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting
cp "/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data/gdp.csv" \
   "/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data/fred_md.csv" \
   "/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data/fred_qd_X.csv" \
   "/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data/filled_md.csv" \
   "/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data/filled_qd.csv" \
   data/
/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python generate_schema.py
rm data/gdp.csv data/fred_md.csv data/fred_qd_X.csv data/filled_md.csv data/filled_qd.csv
```
(The final `rm` restores `data/` to only tracking `bic_lags.csv`, per `.gitignore`'s `*.csv` / `!data/bic_lags.csv` rule — these are working files, not meant to be committed.)

- [ ] **Step 6: Run the verification script again**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_schema.py`
Expected: `ALL SCHEMA CHECKS PASSED`

- [ ] **Step 7: Commit**

```bash
git add generate_schema.py database/schema.sql
git commit -m "Fix schema generator: correct dm_test types/PK, add evaluation and rmse tables"
```

---

## Task 6: Frontend — add a backward-looking nowcast quarter

**Files:**
- Modify: `app.py:44-49` (`QUARTERS`)
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_quarters.py`

**Interfaces:**
- Produces: `QUARTERS: list[str]` with 3 entries (current, previous, previous-previous) instead of 2.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_quarters.py`:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")
import importlib.util

spec = importlib.util.spec_from_file_location("app_module", "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/app.py")
# app.py has Shiny app-construction side effects at import time in some layouts;
# instead, just exec the QUARTERS-relevant lines directly to avoid needing a live Supabase/Shiny context.
import calendar
from datetime import date
from dateutil.relativedelta import relativedelta

exec(compile(
    open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/app.py").read().split("MODEL_DB_NAMES")[0],
    "app_head", "exec"
))

assert len(QUARTERS) == 3, f"expected 3 quarters, got {len(QUARTERS)}: {QUARTERS}"
assert len(set(QUARTERS)) == 3, f"expected 3 distinct quarters, got duplicates: {QUARTERS}"
print("QUARTERS:", QUARTERS)
print("ALL QUARTERS CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_quarters.py`
Expected: `AssertionError: expected 3 quarters, got 2`

- [ ] **Step 3: Add the import and extend `QUARTERS`**

In `app.py`, add to the imports near the top (after `from datetime import date` at line 8):
```python
from dateutil.relativedelta import relativedelta
```

Change `QUARTERS` (lines 46-49) from:
```python
QUARTERS = [
    date_to_quarter(date.today())["current_quarter"],
    date_to_quarter(date.today())["previous_quarter"],
]
```
to:
```python
QUARTERS = [
    date_to_quarter(date.today())["current_quarter"],
    date_to_quarter(date.today())["previous_quarter"],
    date_to_quarter(date.today() - relativedelta(months=3))["previous_quarter"],
]
```

- [ ] **Step 4: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL QUARTERS CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "Add a third (2-quarters-back) nowcast option to the quarter selector"
```

---

## Task 7: Frontend — DM test display redesign (t-stat + direction, drop the matrix)

**Files:**
- Modify: `pipeline/fetch_functions.py:1-6` (imports), `pipeline/fetch_functions.py:216-251` (`fetch_dm`)
- Modify: `app.py:757-860` (`dm_overlay`)
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_fetch_dm.py`

**Interfaces:**
- Produces: `fetch_dm(models: list[str], flash_month: int) -> list[dict]`, where each dict has keys `model_1`, `model_2`, `test_statistic`, `p_value` — one entry per unique pair among `models`, using whichever `(model_1, model_2)` ordering the `dm_test` table stores (winner-first: negative `test_statistic` favours `model_1`). This replaces the old symmetric `dict[tuple[str,str], float|None]` p-value matrix.

- [ ] **Step 1: Write the failing verification script**

This one can't hit live Supabase (paused), so it verifies the *shape* of the new contract using a stub in place of the real client — confirming the function signature/return-type changed as intended, not the live values:

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_fetch_dm.py`:

```python
import sys, inspect
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")
from pipeline.fetch_functions import fetch_dm

sig = inspect.signature(fetch_dm)
return_annotation = str(sig.return_annotation)
assert "list" in return_annotation.lower() or "List" in return_annotation, \
    f"fetch_dm must return a list of pairwise dicts, not a matrix. Got annotation: {return_annotation}"

source = inspect.getsource(fetch_dm)
assert "combinations" in source, "fetch_dm should iterate unique pairs via itertools.combinations, not a full m1×m2 grid"
assert "test_statistic" in source, "fetch_dm must select test_statistic, not just p_value"

print("ALL FETCH_DM SHAPE CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_fetch_dm.py`
Expected: `AssertionError: fetch_dm must return a list of pairwise dicts...`

- [ ] **Step 3: Rewrite `fetch_dm` in `pipeline/fetch_functions.py`**

Add to the imports at the top of the file (after line 4, `from datetime import date`):
```python
from itertools import combinations
```

Replace the whole `fetch_dm` function (lines 216-251) with:
```python
######## Function 5: Fetch Evaluation Metrics (DM) ########
## Fetches pairwise DM test statistics for all unique pairs among `models` ##
def fetch_dm(models: list[str], flash_month: int) -> list[dict]:
    """
    Returns one row per unique pair of models: {model_1, model_2,
    test_statistic, p_value}. Uses whichever (model_1, model_2) ordering
    the dm_test table stores for that pair — winner-first, i.e. a negative
    test_statistic always favours model_1.
    """
    results = []
    for m1, m2 in combinations(models, 2):
        result = supabase.table("dm_test") \
            .select("model_1", "model_2", "test_statistic", "p_value") \
            .eq("model_1", m1).eq("model_2", m2).eq("version", flash_month) \
            .execute()
        if not result.data:
            result = supabase.table("dm_test") \
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

- [ ] **Step 4: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL FETCH_DM SHAPE CHECKS PASSED`

- [ ] **Step 5: Replace the matrix rendering in `app.py`'s `dm_overlay`**

In `app.py`, replace the block from `matrix = fetch_dm(db_models,flash_month)` through the `dm_table = ...` assignment (lines 767-801) with:
```python
        dm_pairs = fetch_dm(db_models, flash_month)
        metrics = fetch_rmse(db_models)

        pair_rows = []
        for pair in dm_pairs:
            if pair["test_statistic"] is None:
                continue
            name_1 = from_db_name(pair["model_1"])
            name_2 = from_db_name(pair["model_2"])
            stat = pair["test_statistic"]
            favored = name_1 if stat <= 0 else name_2
            pair_rows.append(
                ui.p(
                    f"{name_1} − {name_2}:  t = {stat:+.2f}  (favors {favored})",
                    style=f"color: {t['text_primary']}; margin: 0.5rem 0;",
                )
            )

        dm_list = (
            ui.div(*pair_rows)
            if pair_rows
            else ui.p("No DM comparisons available for the selected models.",
                      style=f"color: {t['text_secondary']};")
        )
```

Then change the "Two columns" block (lines 838-849) from:
```python
                ui.div(
                    ui.div(dm_table, style="flex: 1; overflow-x: auto;"),
                    ui.div(
                        *rmse_lines,
                        style=(
                            f"min-width: 160px; padding-left: 2rem; "
                            f"border-left: 1px solid {t['border']}; margin-left: 1.5rem;"
                        ),
                    ),
                    style="display: flex; align-items: flex-start;",
                ),
```
to:
```python
                ui.div(
                    ui.div(dm_list, style="flex: 1;"),
                    ui.div(
                        *rmse_lines,
                        style=(
                            f"min-width: 160px; padding-left: 2rem; "
                            f"border-left: 1px solid {t['border']}; margin-left: 1.5rem;"
                        ),
                    ),
                    style="display: flex; align-items: flex-start;",
                ),
```

Also update the explanatory paragraph above it (line 834-837) — it currently says "DM test p-value > significance level of 0.1" — change to reference the t-statistic:
```python
                ui.p(
                    "The Diebold-Mariano test checks whether one model's forecast errors are significantly smaller than another's. A test statistic further from zero than ±1.96 (5%) or ±1.64 (10%) indicates a significant difference; the sign shows which model is favored.",
                    style=f"color: {t['text_secondary']}; margin-bottom: 1.25rem;",
                ),
```

- [ ] **Step 6: Smoke-check the file still parses**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python -c "import ast; ast.parse(open('/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/app.py').read())"`
Expected: no output (no `SyntaxError`).

- [ ] **Step 7: Commit**

```bash
git add pipeline/fetch_functions.py app.py
git commit -m "Replace DM p-value matrix with a single t-stat + direction per model pair"
```

---

## Task 8: Frontend — fix model and ensemble descriptions

**Files:**
- Modify: `app.py:68-73` (`MODEL_DESCRIPTIONS`)
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_descriptions.py`

**Interfaces:**
- Produces: `MODEL_DESCRIPTIONS: dict[str, str]` — same keys, corrected text for `"Ensemble"` and `"RF Lags UMIDAS"`.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_descriptions.py`:

```python
source = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/app.py").read()

assert "combines predictions from all other models" not in source, \
    "ensemble description must not say 'all other models' (misleading vs. actual 5-model composition)"
assert "5 backend models" in source or "5 models" in source, \
    "ensemble description must state the actual model count (5)"
assert "quarterly averages as features" not in source.split('"RF Lags UMIDAS"')[1].split("}")[0], \
    "RF Lags UMIDAS description must not claim quarterly averages (it uses month-of-quarter U-MIDAS columns)"

print("ALL DESCRIPTION CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_descriptions.py`
Expected: `AssertionError: ensemble description must not say 'all other models'...`

- [ ] **Step 3: Fix `MODEL_DESCRIPTIONS` in `app.py`**

Replace lines 68-73:
```python
MODEL_DESCRIPTIONS = {
    "Ensemble": "An ensemble model that combines predictions from all other models.",
    "RF Lags Avg": "A Random Forest Bridge Equation model using simple quarterly averages of monthly data. Includes lags of the quarterly averages as features.",
    "RF Lags UMIDAS": "A Random Forest using U-MIDAS to treat each monthly observation as a distinct input. Includes lags of the quarterly averages as features.",
    "LASSO UMIDAS": "A Regularized U-MIDAS regression using monthly variables from the current quarter only.",
}
```
with:
```python
MODEL_DESCRIPTIONS = {
    "Ensemble": "A simple average of 5 backend models: 3 LASSO variants (simple average, simple average + lags, U-MIDAS) and 2 Random Forest variants (simple average + lags, U-MIDAS + lags). Only 3 of these 5 models are shown individually in this app.",
    "RF Lags Avg": "A Random Forest Bridge Equation model using simple quarterly averages of monthly data. Includes lags of the quarterly averages as features.",
    "RF Lags UMIDAS": "A Random Forest using U-MIDAS to treat each month within the quarter as a separate input, not a quarterly average. Includes quarterly lags of both the monthly U-MIDAS block and the quarterly variables as features.",
    "LASSO UMIDAS": "A Regularized U-MIDAS regression using monthly variables from the current quarter only.",
}
```

- [ ] **Step 4: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL DESCRIPTION CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "Fix ensemble and RF-UMIDAS model descriptions to match actual behavior"
```

---

## Task 9: Doc — shared python-docx edit helpers + citation/LASSO-claim fixes

**Files:**
- Create: `scripts/docx_edit_helpers.py`
- Modify: `Technical Documentation.docx` (via script, run once, output committed)
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task9.py`

**Interfaces:**
- Produces: `scripts/docx_edit_helpers.py` exposing `replace_paragraph_text(doc, para_index: int, old: str, new: str) -> None` (asserts `old` is found in that paragraph's text, then does a run-preserving replace: clears the paragraph's runs and adds one run with the new text — acceptable here since these are plain-text corrections, not formatted spans) and `find_paragraph_index(doc, substring: str) -> int` (returns the first paragraph index containing `substring`, raises if not found or if found more than once).

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task9.py`:

```python
import docx
d = docx.Document("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx")
full_text = "\n".join(p.text for p in d.paragraphs)

assert "fetch_evaluation_metrics" not in full_text, "stale function name citation still present"
assert "fetch_rmse retrieves RMSE" in full_text, "corrected citation not found"

assert "too many variables for it to be computationally efficient" not in full_text, \
    "false LASSO efficiency claim still present"
assert "hdmpy" in full_text, "replacement text should name the actual bottleneck (hdmpy implementation)"

print("ALL TASK 9 DOC CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task9.py`
Expected: `AssertionError: stale function name citation still present`

- [ ] **Step 3: Create `scripts/docx_edit_helpers.py`**

```python
"""Shared helpers for scripted, targeted edits to Technical Documentation.docx."""


def find_paragraph_index(doc, substring: str) -> int:
    matches = [i for i, p in enumerate(doc.paragraphs) if substring in p.text]
    if not matches:
        raise ValueError(f"No paragraph contains: {substring!r}")
    if len(matches) > 1:
        raise ValueError(f"Substring is ambiguous, found in paragraphs {matches}: {substring!r}")
    return matches[0]


def replace_paragraph_text(doc, para_index: int, old: str, new: str) -> None:
    """Replace `old` with `new` within a single paragraph's text, preserving
    the paragraph's style but not per-run formatting (acceptable for these
    plain-body-text corrections)."""
    para = doc.paragraphs[para_index]
    if old not in para.text:
        raise ValueError(f"Paragraph {para_index} does not contain: {old!r}")
    new_text = para.text.replace(old, new)
    for run in list(para.runs):
        run.text = ""
    if para.runs:
        para.runs[0].text = new_text
    else:
        para.add_run(new_text)


def remove_paragraph(doc, para_index: int) -> None:
    para = doc.paragraphs[para_index]
    para._element.getparent().remove(para._element)
```

- [ ] **Step 4: Write and run the one-off edit script for Task 9**

Create and run `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/apply_doc_task9.py`:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")
import docx
from scripts.docx_edit_helpers import find_paragraph_index, replace_paragraph_text

PATH = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx"
doc = docx.Document(PATH)

idx = find_paragraph_index(doc, "fetch_evaluation_metrics retrieves RMSE")
replace_paragraph_text(
    doc, idx,
    old="fetch_evaluation_metrics retrieves RMSE for selected models",
    new="fetch_rmse retrieves RMSE for selected models",
)

idx = find_paragraph_index(doc, "not computationally efficient")
replace_paragraph_text(
    doc, idx,
    old="LASSO was not run on X4, because there are too many variables for it to be computationally efficient",
    new="LASSO was not run on X4 because our current hdmpy-based implementation does not complete in reasonable time on X4's column count; LASSO itself is used in the literature with far larger predictor sets",
)

doc.save(PATH)
print("Task 9 doc edits applied.")
```

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/apply_doc_task9.py`

- [ ] **Step 5: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL TASK 9 DOC CHECKS PASSED`

- [ ] **Step 6: Commit**

```bash
git add "Technical Documentation.docx" scripts/docx_edit_helpers.py
git commit -m "Fix doc: fetch_rmse citation and false LASSO/X4 efficiency claim"
```

---

## Task 10: Doc — CI Gaussian assumption, COVID caveat, DM squared-loss description

**Files:**
- Modify: `Technical Documentation.docx`
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task10.py`

**Interfaces:**
- Consumes: `scripts/docx_edit_helpers.py` from Task 9.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task10.py`:

```python
import docx
d = docx.Document("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx")
full_text = "\n".join(p.text for p in d.paragraphs)

assert "Gaussian" in full_text, "doc must state the Gaussian-quantile CI assumption explicitly"
assert "tail" in full_text.lower() or "tighter" in full_text.lower(), \
    "doc must caveat that excluding COVID from RMSE likely makes intervals too tight"
assert "Mean Absolute Error (MAE) is used in this case" not in full_text, \
    "stale MAE-based DM description still present"
assert "squared" in full_text.lower(), "DM section must describe squared-loss, matching the code fix"

print("ALL TASK 10 DOC CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task10.py`
Expected: `AssertionError: doc must state the Gaussian-quantile CI assumption explicitly`

- [ ] **Step 3: Write and run the edit script**

Create and run `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/apply_doc_task10.py`:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")
import docx
from scripts.docx_edit_helpers import find_paragraph_index, replace_paragraph_text

PATH = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx"
doc = docx.Document(PATH)

idx = find_paragraph_index(doc, "update_ci_columns() in ci_update.py takes the RMSE values")
replace_paragraph_text(
    doc, idx,
    old="Hence, update_ci_columns() in ci_update.py takes the RMSE values from the rmse Supabase table directly to multiply it with the 50% and 80% critical values to obtain a confidence interval for each model version.",
    new="Hence, update_ci_columns() in ci_update.py takes the RMSE values from the rmse Supabase table directly to multiply it with the 50% and 80% Gaussian critical values (0.674 and 1.282) to obtain a confidence interval for each model version, assuming forecast errors are approximately Normally distributed. Because RMSE excludes the 2020 COVID-19 quarters as outliers (Section 2.3.2), these intervals also exclude that tail-event variance, which likely makes them tighter than the true unconditional uncertainty in GDP growth.",
)

idx = find_paragraph_index(doc, "Mean Absolute Error (MAE) is used in this case")
replace_paragraph_text(
    doc, idx,
    old="Mean Absolute Error (MAE) is used in this case to reduce the impact of big errors.",
    new="Squared error is used in this case, consistent with the RMSE figures reported alongside each DM test.",
)

doc.save(PATH)
print("Task 10 doc edits applied.")
```

- [ ] **Step 4: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL TASK 10 DOC CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add "Technical Documentation.docx"
git commit -m "Doc: state Gaussian CI assumption + COVID tightness caveat, fix DM to squared loss"
```

---

## Task 11: Doc — evaluation period in calendar quarters

**Files:**
- Modify: `Technical Documentation.docx`
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task11.py`

**Interfaces:**
- Consumes: `scripts/docx_edit_helpers.py` from Task 9. `gdp.csv` at `/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data/gdp.csv` to compute the actual calendar range.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task11.py`:

```python
import docx
d = docx.Document("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx")
full_text = "\n".join(p.text for p in d.paragraphs)

idx_text = [p.text for p in d.paragraphs if "162 data points before it make up the train set" in p.text]
assert idx_text, "target paragraph not found"
assert "Q" in idx_text[0] and ("19" in idx_text[0] or "20" in idx_text[0]), \
    "evaluation period must be stated in calendar quarters (e.g. 2001:Q3), not just observation counts"

print("ALL TASK 11 DOC CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task11.py`
Expected: `AssertionError: evaluation period must be stated in calendar quarters...`

- [ ] **Step 3: Write and run the edit script — it computes the calendar range itself from `gdp.csv`, no manual value-filling**

Create and run `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/apply_doc_task11.py`:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")
import docx
import pandas as pd
from scripts.docx_edit_helpers import find_paragraph_index, replace_paragraph_text

DATA_DIR = "/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data"
TEST_SIZE, TRAIN_SIZE = 100, 162  # must match pipeline/poos.py's TEST_SIZE/TRAIN_SIZE constants

gdp = pd.read_csv(f"{DATA_DIR}/gdp.csv")
gdp["sasdate"] = pd.to_datetime(gdp["sasdate"])
known = gdp.dropna(subset=["GDPC1_t"])
test_quarters = known["sasdate"].iloc[-TEST_SIZE:]
test_start = test_quarters.iloc[0].to_period("Q")
test_end = test_quarters.iloc[-1].to_period("Q")
train_start_idx = max(0, len(known) - TEST_SIZE - TRAIN_SIZE)
train_start = known["sasdate"].iloc[train_start_idx].to_period("Q")

PATH = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx"
doc = docx.Document(PATH)

idx = find_paragraph_index(doc, "162 data points before it make up the train set")
old = doc.paragraphs[idx].text
new = old + (
    f" In calendar terms, the 100-quarter test period runs from {test_start} "
    f"to {test_end}, with training data drawn from as early as {train_start}."
)
replace_paragraph_text(doc, idx, old=old, new=new)

doc.save(PATH)
print(f"Task 11 doc edits applied. Test period: {test_start}-{test_end}, training from {train_start}.")
```

- [ ] **Step 4: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL TASK 11 DOC CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add "Technical Documentation.docx"
git commit -m "Doc: state POOS evaluation period in calendar quarters"
```

---

## Task 12: Doc — remove the redundant "mean RMSE across models per version" table

**Files:**
- Modify: `Technical Documentation.docx`
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task12.py`

**Interfaces:**
- Consumes: `scripts/docx_edit_helpers.py` from Task 9.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task12.py`:

```python
import docx
d = docx.Document("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx")
full_text = "\n".join(p.text for p in d.paragraphs)

assert "Table 4: Mean RMSE across all models" not in full_text, \
    "redundant per-version mean-RMSE table caption should be removed"
table_texts = ["\n".join(c.text for row in t.rows for c in row.cells) for t in d.tables]
assert not any("Mean RMSE across all models" in tt for tt in table_texts), \
    "the mean-RMSE-per-version table itself should be removed, not just its caption"
assert "Table 5" in full_text, "text should point the reader to Table 5 instead"

print("ALL TASK 12 DOC CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task12.py`
Expected: `AssertionError: redundant per-version mean-RMSE table caption should be removed`

- [ ] **Step 3: Write and run the edit script**

Create and run `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/apply_doc_task12.py`:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")
import docx
from scripts.docx_edit_helpers import find_paragraph_index, replace_paragraph_text, remove_paragraph

PATH = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx"
doc = docx.Document(PATH)

# Rewrite the surrounding text to point at Table 5 instead of the table being removed
idx = find_paragraph_index(doc, "Comparing the mean RMSE of each version")
old = doc.paragraphs[idx].text
new = old.replace("(see Table 4)", "(visible directly in Table 5, which reports RMSE by model across all 6 versions)")
replace_paragraph_text(doc, idx, old=old, new=new)

# Remove the "Table 4: Mean RMSE across all models" caption paragraph
idx = find_paragraph_index(doc, "Table 4: Mean RMSE across all models")
remove_paragraph(doc, idx)

# Remove the table object itself (the 6-row Version/Mean-RMSE table)
for table in doc.tables:
    header = [c.text for c in table.rows[0].cells]
    if header == ["Version", "Mean RMSE across all models"]:
        table._element.getparent().remove(table._element)
        break
else:
    raise ValueError("Could not find the Version/Mean-RMSE-across-all-models table to remove")

doc.save(PATH)
print("Task 12 doc edits applied.")
```

- [ ] **Step 4: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL TASK 12 DOC CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add "Technical Documentation.docx"
git commit -m "Doc: remove redundant mean-RMSE-per-version table, point to Table 5 instead"
```

---

## Task 13: Doc — add page numbers

**Files:**
- Modify: `Technical Documentation.docx`
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task13.py`

**Interfaces:**
- None beyond the docx file itself.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task13.py`:

```python
import zipfile

PATH = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx"
with zipfile.ZipFile(PATH) as z:
    footer_files = [n for n in z.namelist() if n.startswith("word/footer")]
    assert footer_files, "no footer parts found in the docx"
    found_page_field = False
    for name in footer_files:
        xml = z.read(name).decode("utf-8", errors="ignore")
        if 'w:fldSimple' in xml and 'PAGE' in xml or 'instrText' in xml and 'PAGE' in xml:
            found_page_field = True
    assert found_page_field, "no PAGE field found in any footer part"

print("ALL TASK 13 DOC CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task13.py`
Expected: `AssertionError: no PAGE field found in any footer part` (footers are currently empty, confirmed during investigation).

- [ ] **Step 3: Write and run the edit script**

Create and run `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/apply_doc_task13.py`:

```python
import docx
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PATH = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx"
doc = docx.Document(PATH)

def add_page_number_field(paragraph):
    paragraph.alignment = 1  # center
    run = paragraph.add_run()
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_begin)
    run._r.append(instr_text)
    run._r.append(fld_char_end)

for section in doc.sections:
    footer = section.footer
    footer.is_linked_to_previous = False
    para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    add_page_number_field(para)

doc.save(PATH)
print("Task 13 doc edits applied.")
```

- [ ] **Step 4: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL TASK 13 DOC CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add "Technical Documentation.docx"
git commit -m "Doc: add page numbers to every section's footer"
```

---

## Task 14: Doc — remove the Lorem-ipsum DM screenshot mockup

**Files:**
- Modify: `Technical Documentation.docx`
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task14.py`

**Interfaces:**
- None beyond the docx file itself. A real screenshot of the redesigned DM UI (Task 7) cannot be captured in this pass — Supabase is paused and restoring it is explicitly out of scope (per the design spec) — so this task removes the misleading mockup and leaves a note, rather than fabricating or leaving a stale image in place.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task14.py`:

```python
import zipfile
import docx

PATH = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx"

with zipfile.ZipFile(PATH) as z:
    media = [n for n in z.namelist() if n.startswith("word/media/")]
    assert "word/media/image7.png" not in media, \
        "the Lorem-ipsum DM screenshot mockup (image7.png) should be removed"

d = docx.Document(PATH)
full_text = "\n".join(p.text for p in d.paragraphs)
assert "pending" in full_text.lower() and "screenshot" in full_text.lower(), \
    "a note should explain the screenshot is pending a fresh capture once Supabase is restored"

print("ALL TASK 14 DOC CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task14.py`
Expected: `AssertionError: the Lorem-ipsum DM screenshot mockup (image7.png) should be removed`

- [ ] **Step 3: Locate and remove the image's paragraph, insert a note**

Create and run `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/apply_doc_task14.py`:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")
import docx
from docx.oxml.ns import qn

PATH = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx"
doc = docx.Document(PATH)

# Find the paragraph containing the inline image tied to image7.png's relationship id
target_rid = None
for rel_id, rel in doc.part.rels.items():
    if rel.target_ref.endswith("image7.png"):
        target_rid = rel_id
        break
assert target_rid is not None, "could not find image7.png's relationship id"

target_para = None
for para in doc.paragraphs:
    blips = para._element.findall(".//" + qn("a:blip"))
    for blip in blips:
        if blip.get(qn("r:embed")) == target_rid:
            target_para = para
            break
    if target_para is not None:
        break
assert target_para is not None, "could not find the paragraph containing the DM screenshot image"

target_para.text = (
    "[Screenshot pending: this figure will show the redesigned DM Statistics "
    "panel (single t-statistic per model pair, no matrix) once Supabase is "
    "restored and the live app can be captured. The previous version of this "
    "figure was an early mockup with placeholder Lorem ipsum text and has "
    "been removed.]"
)

doc.save(PATH)
print("Task 14 doc edits applied.")
```

- [ ] **Step 4: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL TASK 14 DOC CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add "Technical Documentation.docx"
git commit -m "Doc: remove Lorem-ipsum DM screenshot mockup, note real capture is pending"
```

---

## Task 15: Doc — sync AR benchmark section + Table 1 with the code changes

**Files:**
- Modify: `Technical Documentation.docx`
- Test: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task15.py`

**Interfaces:**
- Consumes: `scripts/docx_edit_helpers.py` from Task 9. Describes the AR benchmark implementation from Task 1.

- [ ] **Step 1: Write the failing verification script**

Create `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task15.py`:

```python
import docx
d = docx.Document("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx")
full_text = "\n".join(p.text for p in d.paragraphs)

assert "direct" in full_text.lower() and "2-step" in full_text.lower(), \
    "AR benchmark section must describe the direct 2-step forecast fallback"
assert "PCECC96" in full_text and "lag" in full_text.lower(), \
    "doc should reflect that GDP components enter as lags only, not contemporaneously"
assert "pending" in full_text.lower() and "re-run" in full_text.lower(), \
    "Results section must caveat that RMSE/DM tables predate this pass's methodology fixes"

print("ALL TASK 15 DOC CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/verify_doc_task15.py`
Expected: `AssertionError: AR benchmark section must describe the direct 2-step forecast fallback`

- [ ] **Step 3: Write and run the edit script**

Create and run `/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/apply_doc_task15.py`:

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting")
import docx
from scripts.docx_edit_helpers import find_paragraph_index, replace_paragraph_text

PATH = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/Technical Documentation.docx"
doc = docx.Document(PATH)

idx = find_paragraph_index(doc, "The AR(2) model is implemented via Ordinary Least Squares")
old = doc.paragraphs[idx].text
new = old + (
    " When the immediately preceding quarter's GDP is not yet released, the "
    "benchmark uses a direct 2-step forecast instead — regressing GDP growth "
    "on the t-2 and t-3 lags directly, rather than imputing the missing t-1 "
    "lag from another model's prediction. This keeps the benchmark a genuine "
    "floor: it only ever uses real, released GDP history."
)
replace_paragraph_text(doc, idx, old=old, new=new)

idx = find_paragraph_index(doc, "We also removed series that are either highly correlated with GDP")
old = doc.paragraphs[idx].text
new = old + (
    " Real GDP components that remain in the dataset — PCECC96 (consumption), "
    "GPDIC1 (investment), and EXPGSC1 (exports) — are restricted to entering "
    "only as lags, never contemporaneously, for the same reason: using them "
    "same-quarter would let a model partially reconstruct GDP from its own "
    "components rather than genuinely forecast it."
)
replace_paragraph_text(doc, idx, old=old, new=new)

idx = find_paragraph_index(doc, "In order to compare our 6 models, we use three different measures")
old = doc.paragraphs[idx].text
new = old + (
    " Note: the RMSE and DM figures in this section (Tables 5-9, Appendix "
    "Tables B-D) predate the AR-benchmark, GDP-lag, and DM-loss-function "
    "fixes described above (Sections 2.2.1, 2.1.1, 2.3.3) — they are pending "
    "a re-run of the full historical POOS pipeline and will be refreshed in "
    "a follow-up update, not fabricated or hand-edited here."
)
replace_paragraph_text(doc, idx, old=old, new=new)

doc.save(PATH)
print("Task 15 doc edits applied.")
```

- [ ] **Step 4: Run the verification script again**

Run: same command as Step 2.
Expected: `ALL TASK 15 DOC CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add "Technical Documentation.docx"
git commit -m "Doc: describe AR benchmark direct-forecast fallback and GDP-component lag-only rule"
```

---

## Final check: full-repo smoke test

- [ ] **Step 1: Confirm every modified Python file still imports cleanly**

Run:
```bash
/private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python -c "
import sys
sys.path.insert(0, '/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting')
import ast
for f in [
    'pipeline/output_x.py', 'pipeline/output_x_poos.py', 'pipeline/poos.py',
    'pipeline/dm_test.py', 'pipeline/ragged_edge.py', 'pipeline/fetch_functions.py',
    'generate_schema.py', 'app.py', 'scripts/docx_edit_helpers.py',
]:
    ast.parse(open(f'/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/{f}').read())
    print(f'{f}: OK')
"
```
Expected: `OK` printed for every file, no `SyntaxError`.

- [ ] **Step 2: Re-run every verification script from Tasks 1-15 in sequence**

Run:
```bash
for f in verify_ar_benchmark verify_dm_loss verify_gdp_components verify_ragged_edge \
         verify_schema verify_quarters verify_fetch_dm verify_descriptions \
         verify_doc_task9 verify_doc_task10 verify_doc_task11 verify_doc_task12 \
         verify_doc_task13 verify_doc_task14 verify_doc_task15; do
  echo "=== $f ==="
  /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/venv/bin/python \
    /private/tmp/claude-502/-Users-Jennifur-Desktop-random-projects-DSE6786-Nowcasting/9ea99b5c-cda5-4087-933d-0d012e0aa720/scratchpad/$f.py
done
```
Expected: every script prints its `ALL ... CHECKS PASSED` (or `ALL ... PASSED`) line, no assertion errors.

- [ ] **Step 3: Final commit if anything was left unstaged**

```bash
git status
```
Expected: clean working tree (everything already committed task-by-task above).

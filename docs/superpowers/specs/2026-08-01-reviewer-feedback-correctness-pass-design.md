# Reviewer Feedback Correctness Pass — Design

## Context

The project received detailed reviewer feedback (backend + frontend) covering bugs,
inconsistencies, and misleading descriptions across the pipeline, database schema,
frontend, and the Technical Documentation.docx. This spec covers the **correctness
pass**: fixing everything that is a bug, an internal inconsistency, or a misleading
statement. Genuine methodology *improvements* (as opposed to corrections) are
explicitly deferred — see "Deferred" section.

Grounding: every item below was verified against the actual code, the real FRED
QD/MD data (`/Users/Jennifur/Desktop/NUS/Y3/Y3S2/DSE3101/DSE6786-Nowcasting/data/`),
and the Technical Documentation.docx now living at the project root, not assumed
from the reviewer's prose alone.

The full historical POOS re-run (needed to regenerate correct RMSE/DM numbers for
the doc's tables and the live dashboard) is **out of scope for this pass** — it
takes hours and needs `SUPABASE_SERVICE_KEY`, which the user will run themselves
later. Tables/values that would change are marked with a clear placeholder/note
instead of being left with stale numbers or being fabricated.

## 1. Pipeline bug fixes

### 1.1 AR benchmark rewrite + AR(2)/AR(4) consistency bug

**Bug found**: `make_build_X()` in `pipeline/output_x_poos.py:190-213` takes its
own `n_lags` parameter (default 4, meant for the X1–X4 feature-matrix builders)
and passes it straight into `build_X_AR_from_cut(gdp_cut, gdp_actual, n_lags)`
(line 213), silently overriding that function's own `n_lags: int = 2` default
(line 153). Since `pipeline/historical.py` calls `make_build_X(build_name)` with
no explicit `n_lags` override, POOS evaluation trains an AR(4) benchmark while the
live nowcast path (`pipeline/output_x.py:340`, `build_X_AR(n_lags=2)`) produces
AR(2) forecasts. Two different specs by accident, not by design — matches the
doc's own claim of "AR(2) model, 2 lags... via AIC" (§2.2.1).

**Also**: the current AR benchmark isn't a minimal benchmark. Per
`pipeline/prediction.py` / `pipeline/historical.py` MODEL_REGISTRY, the AR
benchmark's first lag gets filled from the ensemble ("All_Model_Average")
nowcast when unavailable, before running AR(2) OLS — i.e. the "benchmark"
partially depends on the fancy machinery it's supposed to be a floor against.

**Fix**:
- Rewrite `pipeline/models/AR_benchmark.py`'s `ar_model_nowcast` (or a sibling
  function used consistently everywhere) to do a **direct multi-step forecast**:
  - If yₜ₋₁ (the immediately preceding quarter's GDP) is available: standard
    1-step AR(2), OLS on yₜ₋₁, yₜ₋₂.
  - If yₜ₋₁ is NOT yet released: direct 2-step forecast — OLS of yₜ on yₜ₋₂, yₜ₋₃
    (skipping the unavailable lag entirely, no ensemble fill-in).
- Give the AR path in `make_build_X` / `build_X_AR_from_cut` its own dedicated
  lag-count parameter, decoupled from the `n_lags` used for X1–X4, so this class
  of bug can't recur. Target spec: **AR(2) everywhere** (live nowcast path via
  `output_x.py::build_X_AR`, and POOS evaluation via
  `output_x_poos.py::build_X_AR_from_cut`), both driven by the same
  `AR_benchmark.py` implementation.
- Update `pipeline/prediction.py` and `pipeline/historical.py` call sites
  accordingly — remove the ensemble-fill-first-lag logic.

### 1.2 DM test loss function mismatch

**Bug found**: `pipeline/dm_test.py:100`, `dm_test()` defaults `loss="absolute"`.
Nothing overrides this in `compare_model_pairs()` or `main()`, so the DM
statistics pushed to Supabase (and shown on the frontend) test for a difference
in MAE while the accompanying RMSE values are squared-loss. The doc confirms
this was intentional ("Mean Absolute Error (MAE) is used in this case to reduce
the impact of big errors", §2.3.3) — but it's methodologically inconsistent: if
RMSE is reported, DM must use squared loss on the same errors, or the two
numbers answer different questions.

**Fix**: default `loss="squared"` in `dm_test()`, and make sure
`compare_model_pairs()` doesn't override it back to absolute.

### 1.3 GDP components: lag-only treatment

**Inconsistency found**: `pipeline/load_data.py:85-92` drops OUTNFB, OUTBS,
OPHPBS because they're "related to GDP" (near-tautological reconstruction), but
PCECC96 (consumption), GPDIC1 (investment), and EXPGSC1 (exports) — actual GDP
components — are kept and enter contemporaneously in the X1–X4 feature matrices
built in `pipeline/output_x.py` / `pipeline/output_x_poos.py`. This is why
Version 6 (near-complete current-quarter data) looks unrealistically good — it's
close to reconstructing GDP from its own components.

**Fix**: PCECC96, GPDIC1, EXPGSC1 drop out of the **contemporaneous (lag-0)**
block of the feature matrix but keep their lagged versions, exactly like GDP's
own lags. Concretely: in `build_X1`/`build_X2`/`build_X3`/`build_X4` (and their
`_from_cut` POOS twins), exclude these three columns from the un-lagged
`df_q`/`qd1` block before `_add_lags`/`_add_lags_df` runs, while still letting
`_add_lags` produce their `_lag1`...`_lagN` columns from the original series.

This will make Version 6 results look less impressive — that's expected and
correct per the reviewer.

### 1.4 Ragged-edge fillna(0) hardening

**Verified**: on the real `fred_qd_X.csv`/`fred_md.csv` data, every one of the
117/115 `bic_lags.csv` variables has **zero leftover NaNs** after
`_fill_series`'s interpolate+AR-forecast step — raw data only has 1-3 trailing
NaNs per column (unreleased recent months), all resolved by the fill.
`fillna(0)` in `pipeline/ragged_edge.py:137,168-169` is currently a **no-op** on
this dataset, consistent with the user's memory that problem variables were
already trimmed in an earlier pass.

**Fix (hardening, not a live-bug fix)**: replace the blind `fillna(0)` with an
explicit check that raises/logs loudly if any NaN survives the fill step,
turning a silent landmine into a documented, enforced invariant. Add a comment
explaining why (so a future variable added to `bic_lags.csv` with real leading
gaps fails loudly instead of being silently zeroed into training data).

### 1.5 schema.sql / generate_schema.py

**Bugs found** in `database/schema.sql:571-578` (`dm_test` table): `model_1`/
`model_2` declared `NUMERIC` but hold strings like `"AR_Benchmark_v1"`, and
`PRIMARY KEY (sasdate)` references a column that isn't even in the table (would
be rejected by Postgres on a clean run). Root cause traced to
`generate_schema.py`: it reads columns from `data/dm_pval_matrix.csv` (a stale,
unrelated file) rather than the actual `dm_test` table structure that
`pipeline/dm_test.py::push_dm_results_to_supabase` writes to (`version`,
`model_1`, `model_2`, `test_statistic`, `p_value`, upsert conflict key
`version,model_1,model_2`), and its `get_sql_type()`/`generate_create_table()`
helpers blindly type every non-`sasdate` column `NUMERIC` and always emit
`PRIMARY KEY (sasdate)` regardless of the table. Also confirmed: `rmse` and
`evaluation` tables (referenced by `fetch_rmse()` and actively written by
`pipeline/evaluation_table_hist.py`) don't appear in `CSV_FILES` at all, so
they're entirely missing from schema.sql.

**Fix**: fix the generator, not just its output (schema.sql's own header says
"Do not edit manually. Re-run generate_schema.py to update.") —
- Add a `dm_test` table generator with correct types (`version` NUMERIC,
  `model_1`/`model_2` TEXT, `test_statistic`/`p_value` NUMERIC) and
  `PRIMARY KEY (version, model_1, model_2)`, matching the upsert conflict key.
- Add `evaluation` table generator: `quarter_date` DATE, `version` NUMERIC,
  `month_date` DATE, `gdp_actual` NUMERIC, one NUMERIC column per model
  (`AR_Benchmark`, `RF_Lags_Average`, `RF_Lags_UMIDAS`, `LASSO_UMIDAS`,
  `LASSO_Average`, `LASSO_Lags_Average`, `All_Model_Average`), unique/PK on
  `(quarter_date, version)` — matches `push_forecasts_to_evaluation()`.
- Add `rmse` table generator: `model` TEXT, `version` NUMERIC, `rmse` NUMERIC,
  PK on `(model, version)` — matches `calculate_and_upsert_rmse()` /
  `calculate_mean_rmse_by_model()`.
- Regenerate `database/schema.sql` from the fixed generator.
- No live Supabase connection needed for this — everything is derived from
  what the code actually writes.

## 2. Frontend fixes (`app.py`)

- **Add a backward-looking quarter**: extend the quarter selector to include one
  more historical quarter (e.g. 2025:Q4) alongside current/previous, matching
  the NY Fed's practice of showing nowcasts for quarters that are still not
  fully certain.
- **DM test display redesign**: replace the symmetric p-value matrix with a
  single DM t-statistic and an explicit "(Model 1 − Model 2)" direction
  statement; keep RMSE displayed alongside for reference. Uses the corrected
  squared-loss DM values from §1.2.
- **Model description fixes**: audit `MODEL_DESCRIPTIONS` in `app.py` —
  "RF Lags UMIDAS" currently claims "quarterly averages as features" when the
  actual X4 design matrix uses month-of-quarter U-MIDAS columns, not quarterly
  averages. Correct this and check the other three descriptions for similar
  imprecision.
- **Ensemble description fix**: `MODEL_DESCRIPTIONS["Ensemble"]` currently says
  "combines predictions from all other models," which reads as "the 3 models
  shown in this app." Actual composition (confirmed via the doc and
  `historical.py`/`prediction.py`) is 5 backend models — LASSO×3 + RF×2,
  excluding AR_Benchmark. Rewrite the description to state this explicitly
  (name the 5 or state the count), without changing the actual computation.

## 3. Documentation fixes (`Technical Documentation.docx`)

- §1.2.2: fix citation — `fetch_evaluation_metrics` → `fetch_rmse`.
- §2.2.2.3: remove "LASSO was not run on X4 because it is not computationally
  efficient" — replace with an accurate statement (the current `hdmpy`-based
  implementation is slow on X4's column count; LASSO itself scales to far
  larger predictor sets in the literature).
- §2.3.2: state explicitly that prediction intervals assume Gaussian
  quantiles (0.674/1.282 critical values × RMSE); add a caveat that excluding
  COVID quarters from RMSE, while reasonable for relative model comparison,
  likely makes intervals too tight by removing tail-event variance from GDP
  growth.
- §2.3.3: update the DM test description to reflect squared loss (matches
  §1.2's code fix) instead of MAE.
- §2.3.1: state the evaluation period in calendar quarters (derived from
  `TEST_SIZE=100`, `TRAIN_SIZE=162`, and the actual GDP series date range),
  not just observation counts.
- §2.3.4.1 / Table 4: remove the "mean RMSE across all models per version"
  table — redundant with Table 5 (best model per version), which already shows
  the same later-versions-are-better trend, per reviewer. Keep the appendix's
  per-model-across-versions summary (feeds the live frontend via
  `calculate_mean_rmse_by_model`, not purely a doc artifact).
- Add page numbers (footer field) — currently absent from all sections.
- Replace the DM Statistics screenshot (confirmed to be a Lorem-ipsum mockup
  showing "Model 1"/"Combined model" placeholders and a symmetric matrix) with
  a real screenshot of the redesigned DM UI from §2, once that UI exists.
- Sync model descriptions, AR benchmark section (§2.2.1), and the dropped-series
  appendix table with the code changes above (§1.1, §1.3, §2).
- Anywhere a table's numbers depend on the deferred historical re-run (§1.1–1.3
  change what the models actually predict), leave a clear inline note
  (e.g. "values pending re-run after AR benchmark / GDP-lag fix") rather than
  stale or fabricated numbers.

## Deferred (explicitly out of scope for this pass)

These are genuine methodology improvements, not corrections — noted in the
doc's Limitations/Future Work section as follow-ups, not implemented now:
- Prediction interval 50%/80% toggle instead of showing both at once, plus
  overlaying two models' intervals for comparison.
- Empirical Gaussianity/coverage validation for prediction intervals (plot
  error distribution vs. Normal, measure empirical coverage in POOS).
- Swapping LASSO's `hdmpy` implementation for a faster equivalent (e.g.
  glmnet-based).
- Changing the ensemble's actual composition (5 vs. 3 models) — only the
  description is being fixed in this pass, not the computation.

## Explicitly out of scope

- The full historical POOS re-run and re-push of corrected RMSE/DM numbers to
  Supabase — user will run this themselves later (needs `SUPABASE_SERVICE_KEY`
  and multiple hours).
- Restoring/recreating the paused Supabase project — not needed for this
  pass's static code/doc edits.

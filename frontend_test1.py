"""
test_fetch_functions.py
-----------------------
Manual integration tests for pipeline/fetch_functions.py.
Each test prints PASS / FAIL and a short summary of what was returned,
so you can eyeball values against what you expect from Supabase.

Run with:
    python test_fetch_functions.py

Requires a .env file with SUPABASE_URL and SUPABASE_KEY set.
"""

from datetime import date
from itertools import combinations
from pipeline.fetch_functions import (
    quarter_to_dates,
    _flash_month_dates,
    _month_end,
    fetch_nowcast_data,
    fetch_confidence_intervals,
    fetch_flash_predictions,
    fetch_historical_data,
    fetch_rmse,
    fetch_dm,
)
from database.client import get_frontend_client

# ---------------------------------------------------------------------------
# Test config -- adjust these to values you know exist in your Supabase DB
# ---------------------------------------------------------------------------

TEST_QUARTER     = "2024:Q4"
TEST_MODEL       = "AR_Benchmark"             # single model in model_forecasts
TEST_MODELS      = ["AR_Benchmark", "LASSO_UMIDAS"]  # 2+ models for RMSE / DM
TEST_START       = date(2020, 1, 1)
TEST_END         = date(2022, 12, 31)
TEST_FLASH_MONTH = 1                          # 1, 2, or 3

# This script is a standalone manual integration test run directly against a
# real Supabase instance (see module docstring), so constructing the client
# once at module scope here is intentional -- it is never imported elsewhere.
client = get_frontend_client()

# ---------------------------------------------------------------------------

PASS = "PASS ✅"
FAIL = "FAIL ❌"

def section(title):
    print("\n" + "-" * 60)
    print("  " + title)
    print("-" * 60)


# ---------------------------------------------------------------------------
# 1. quarter_to_dates
# ---------------------------------------------------------------------------

section("1. quarter_to_dates")
try:
    cases = [
        ("2024:Q1", "2024-03-01"),
        ("2024:Q2", "2024-06-01"),
        ("2024:Q3", "2024-09-01"),
        ("2024:Q4", "2024-12-01"),
    ]
    for q_str, expected in cases:
        result = quarter_to_dates(q_str)
        assert result == expected, f"{q_str}: expected {expected}, got {result}"
        print(f"  {q_str} -> {result}")
    print(f"  {PASS}")
except Exception as e:
    print(f"  {FAIL}: {e}")


# ---------------------------------------------------------------------------
# 2. _month_end
# ---------------------------------------------------------------------------

section("2. _month_end (internal helper)")
try:
    cases = [
        (2024, 1,  "2024-01-31"),
        (2024, 2,  "2024-02-29"),  # 2024 is a leap year
        (2023, 2,  "2023-02-28"),  # 2023 is not
        (2024, 3,  "2024-03-31"),
        (2024, 12, "2024-12-31"),
    ]
    for year, month, expected in cases:
        result = _month_end(year, month)
        assert result == expected, f"_month_end({year},{month}): expected {expected}, got {result}"
        print(f"  _month_end({year}, {month:2d}) -> {result}")
    print(f"  {PASS}")
except Exception as e:
    print(f"  {FAIL}: {e}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 3. _flash_month_dates
# ---------------------------------------------------------------------------

section("3. _flash_month_dates (internal helper)")
try:
    # Each entry should be a (quarter_date, month_date) tuple, both YYYY-MM-DD.
    # In 2024 there are 4 quarters, so we expect 4 pairs per flash_month.
    expected_per_flash = {
        1: [("2024-03-01", "2024-01-31"), ("2024-06-01", "2024-04-30"),
            ("2024-09-01", "2024-07-31"), ("2024-12-01", "2024-10-31")],
        2: [("2024-03-01", "2024-02-29"), ("2024-06-01", "2024-05-31"),
            ("2024-09-01", "2024-08-31"), ("2024-12-01", "2024-11-30")],
        3: [("2024-03-01", "2024-03-31"), ("2024-06-01", "2024-06-30"),
            ("2024-09-01", "2024-09-30"), ("2024-12-01", "2024-12-31")],
    }
    for fm, expected_pairs in expected_per_flash.items():
        pairs = _flash_month_dates(date(2024, 1, 1), date(2024, 12, 31), fm)
        assert pairs == expected_pairs, (
            f"flash_month={fm}: expected {expected_pairs}, got {pairs}"
        )
        print(f"  flash_month={fm}:")
        for qd, md in pairs:
            print(f"    quarter_date={qd}  month_date={md}")
    print(f"  {PASS}")
except Exception as e:
    print(f"  {FAIL}: {e}")


# 4. fetch_nowcast_data
# ---------------------------------------------------------------------------

section("4. fetch_nowcast_data")
try:
    data, month_labels = fetch_nowcast_data(client, TEST_QUARTER)

    assert isinstance(data, dict),         "data should be a dict"
    assert isinstance(month_labels, list), "month_labels should be a list"
    assert len(data) > 0,                  "data should contain at least one model"
    assert len(month_labels) > 0,          "month_labels should not be empty"

    for model, values in data.items():
        assert len(values) == len(month_labels), (
            f"{model}: {len(values)} values but {len(month_labels)} month labels"
        )

    print(f"  quarter     : {TEST_QUARTER}")
    print(f"  month_labels: {month_labels}")
    for model, values in data.items():
        print(f"  {model}: {values}")
    print(f"  {PASS}")
except Exception as e:
    print(f"  {FAIL}: {e}")


# ---------------------------------------------------------------------------
# 5. fetch_confidence_intervals
# ---------------------------------------------------------------------------

section("5. fetch_confidence_intervals")
try:
    month_labels, ci50_lo, ci50_hi, ci80_lo, ci80_hi = fetch_confidence_intervals(
        client, TEST_QUARTER, TEST_MODEL
    )

    assert len(month_labels) > 0, "month_labels should not be empty"
    for name, band in [("ci50_lo", ci50_lo), ("ci50_hi", ci50_hi),
                       ("ci80_lo", ci80_lo), ("ci80_hi", ci80_hi)]:
        assert len(band) == len(month_labels), (
            f"{name}: length {len(band)} != {len(month_labels)}"
        )

    for i in range(len(month_labels)):
        assert ci50_lo[i] <= ci50_hi[i], f"ci50 lower > upper at index {i}"
        assert ci80_lo[i] <= ci80_hi[i], f"ci80 lower > upper at index {i}"
        assert ci80_lo[i] <= ci50_lo[i], f"ci80 lower not wider than ci50 at index {i}"
        assert ci80_hi[i] >= ci50_hi[i], f"ci80 upper not wider than ci50 at index {i}"

    print(f"  quarter    : {TEST_QUARTER}  model: {TEST_MODEL}")
    print(f"  months     : {month_labels}")
    print(f"  ci_50_lower: {ci50_lo}")
    print(f"  ci_50_upper: {ci50_hi}")
    print(f"  ci_80_lower: {ci80_lo}")
    print(f"  ci_80_upper: {ci80_hi}")
    print(f"  {PASS}")
except Exception as e:
    print(f"  {FAIL}: {e}")


# ---------------------------------------------------------------------------
# 6. fetch_flash_predictions
# ---------------------------------------------------------------------------

section("6. fetch_flash_predictions")
try:
    for fm in [1, 2, 3]:
        preds = fetch_flash_predictions(client, TEST_START, TEST_END, fm)
        assert isinstance(preds, dict), "predictions should be a dict"
        assert len(preds) > 0, f"no predictions returned for flash_month={fm}"

        lengths = {m: len(v) for m, v in preds.items()}
        assert len(set(lengths.values())) == 1, (
            f"unequal series lengths for flash_month={fm}: {lengths}"
        )

        n_quarters = list(lengths.values())[0]
        print(f"  flash_month={fm}: {len(preds)} models, {n_quarters} quarters each")
        for model, values in preds.items():
            preview = values[:4]
            suffix = "..." if len(values) > 4 else ""
            print(f"    {model}: {preview}{suffix}")

    print(f"  {PASS}")
except Exception as e:
    print(f"  {FAIL}: {e}")


# ---------------------------------------------------------------------------
# 7. fetch_historical_data
# ---------------------------------------------------------------------------

section("7. fetch_historical_data")
try:
    quarter_labels, actual_values, predictions = fetch_historical_data(
        client, TEST_START, TEST_END, TEST_FLASH_MONTH
    )

    assert isinstance(quarter_labels, list), "quarter_labels should be a list"
    assert isinstance(actual_values,  list), "actual_values should be a list"
    assert isinstance(predictions,    dict), "predictions should be a dict"
    assert len(quarter_labels) > 0,          "quarter_labels should not be empty"
    assert len(quarter_labels) == len(actual_values), (
        f"quarter_labels ({len(quarter_labels)}) and actual_values "
        f"({len(actual_values)}) differ in length"
    )
    for model, values in predictions.items():
        assert len(values) == len(quarter_labels), (
            f"{model}: {len(values)} predictions but {len(quarter_labels)} quarters"
        )

    preview_q = quarter_labels[:4]
    preview_a = actual_values[:4]
    print(f"  range      : {TEST_START} to {TEST_END}  flash_month={TEST_FLASH_MONTH}")
    print(f"  quarters   : {preview_q}{'...' if len(quarter_labels) > 4 else ''}")
    print(f"  actuals    : {preview_a}{'...' if len(actual_values) > 4 else ''}")
    for model, values in predictions.items():
        preview = values[:4]
        suffix = "..." if len(values) > 4 else ""
        print(f"  {model}: {preview}{suffix}")
    print(f"  {PASS}")
except Exception as e:
    print(f"  {FAIL}: {e}")


# ---------------------------------------------------------------------------
# 8. fetch_rmse
# ---------------------------------------------------------------------------

section("8. fetch_rmse")
try:
    metrics = fetch_rmse(client, TEST_MODELS)

    assert isinstance(metrics, dict), "metrics should be a dict"
    for model in TEST_MODELS:
        assert model in metrics,           f"missing model: {model}"
        assert "rmse" in metrics[model],   f"missing 'rmse' key for {model}"
        rmse_val = metrics[model]["rmse"]
        assert isinstance(rmse_val, float), f"rmse for {model} should be a float"
        assert rmse_val > 0,               f"rmse for {model} should be positive"

    print(f"  models: {TEST_MODELS}")
    for model, m in metrics.items():
        print(f"  {model}: mean RMSE = {m['rmse']:.4f}")
    print(f"  {PASS}")
except Exception as e:
    print(f"  {FAIL}: {e}")


# ---------------------------------------------------------------------------
# 9. fetch_dm
# ---------------------------------------------------------------------------

section("9. fetch_dm")
try:
    expected_pairs = {frozenset((m1, m2)) for m1, m2 in combinations(TEST_MODELS, 2)}

    for fm in [1, 2, 3]:
        pairs = fetch_dm(client, TEST_MODELS, fm)

        assert isinstance(pairs, list), "fetch_dm should return a list"

        seen = set()
        for pair in pairs:
            assert isinstance(pair, dict), "each entry should be a dict"
            for key in ("model_1", "model_2", "test_statistic", "p_value"):
                assert key in pair, f"entry missing key {key!r}: {pair}"

            m1, m2 = pair["model_1"], pair["model_2"]
            assert m1 != m2, f"pair should not be a self-comparison: {pair}"
            assert m1 in TEST_MODELS and m2 in TEST_MODELS, (
                f"unexpected model name(s) in pair: {pair}"
            )

            pair_key = frozenset((m1, m2))
            assert pair_key in expected_pairs, f"pair {pair} is not a unique combination of TEST_MODELS"
            assert pair_key not in seen, f"pair {pair} duplicates an already-seen combination"
            seen.add(pair_key)

            stat = pair["test_statistic"]
            assert stat is None or isinstance(stat, float), (
                f"test_statistic for {pair} should be float or None, got {type(stat)}"
            )

            p_value = pair["p_value"]
            assert p_value is None or isinstance(p_value, float), (
                f"p_value for {pair} should be float or None, got {type(p_value)}"
            )
            if p_value is not None:
                assert 0.0 <= p_value <= 1.0, (
                    f"p-value for {pair} = {p_value} is outside [0, 1]"
                )

        print(f"\n  flash_month={fm} DM comparisons ({len(pairs)} pair(s)):")
        for pair in pairs:
            stat = pair["test_statistic"]
            p_value = pair["p_value"]
            if stat is None:
                print(f"    {pair['model_1']} vs {pair['model_2']}: t = --- (no statistic)")
                continue
            favored = pair["model_1"] if stat <= 0 else pair["model_2"]
            p_str = f"{p_value:.4f}" if p_value is not None else "---"
            print(
                f"    {pair['model_1']} vs {pair['model_2']}: "
                f"t = {stat:+.4f}  p = {p_str}  (favors {favored})"
            )

    print(f"\n  {PASS}")
except Exception as e:
    print(f"  {FAIL}: {e}")


# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  All tests complete.")
print("=" * 60 + "\n")
import calendar
from datetime import date
from itertools import combinations

######## Helper Function ########
## Converts quarter string to start and end dates ##
# Needed 'cuz we don't have a column for quarter #
def quarter_to_dates(quarter: str) -> str:
    year, q = quarter.split(":")
    last_month = {"Q1": "03", "Q2": "06", "Q3": "09", "Q4": "12"}[q]
    return f"{int(year)}-{last_month}-01"

def _month_end(year: int, month: int) -> str:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-{last_day}"
######## Function 1: Getting Nowcast Data ########

def fetch_nowcast_data(client, quarter: str) -> dict[str, list[float]]:

    # Step 0: Get input's (quarter) start and end dates:
    quarter_start = quarter_to_dates(quarter)

    # Step 1: Get rows from model_forecasts table for the specified quarter
    result = client.table("model_forecasts") \
        .select("*") \
        .eq("quarter_date", quarter_start) \
        .order("month_date") \
        .execute()

    forecasts = result.data

    # Step 2: Reshape into desired format
    data = {}
    for row in forecasts:
        model = row["model_name"]
        if model not in data:
            data[model] = []
        data[model].append(row["nowcast"])

    # Step 3: Get the month labels
    models_requested = list(data.keys())

    month_labels = [
        row["month_date"] for row in forecasts
        if row["model_name"] == models_requested[0]
    ]

    return data, month_labels


######## Function 3: Fetch Confidence Intervals ########

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

    forecasts = result.data

    # Step 2: Extract each column into its own list
    month_labels = [row["month_date"] for row in forecasts]
    ci_50_lower = [row["ci_50_lb"] for row in forecasts]
    ci_50_upper = [row["ci_50_ub"] for row in forecasts]
    ci_80_lower = [row["ci_80_lb"] for row in forecasts]
    ci_80_upper = [row["ci_80_ub"] for row in forecasts]

    return month_labels, ci_50_lower, ci_50_upper, ci_80_lower, ci_80_upper

######## Helper: quarter_date + month_date pairs for a date range ########
# For each quarter whose quarter_date falls within [start_date, end_date],
# returns a (quarter_date, month_date) pair -- both as YYYY-MM-DD strings.
# Using BOTH fields to filter uniquely identifies one row in model_forecasts,
# avoiding the ambiguity where the same month_date appears under multiple
# quarter_dates (e.g. April is flash month 1 of Q2 AND a revision month for Q1).
#
# Quarter months:
#   Q1: flash 1 = Jan,  flash 2 = Feb,  flash 3 = Mar
#   Q2: flash 1 = Apr,  flash 2 = May,  flash 3 = Jun
#   Q3: flash 1 = Jul,  flash 2 = Aug,  flash 3 = Sep
#   Q4: flash 1 = Oct,  flash 2 = Nov,  flash 3 = Dec

def _flash_month_dates(start_date, end_date, flash_month: int) -> list[tuple[str, str]]:
    start = start_date if isinstance(start_date, date) else date.fromisoformat(str(start_date))
    end   = end_date   if isinstance(end_date,   date) else date.fromisoformat(str(end_date))

    quarter_last_months = [3, 6, 9, 12]

    pairs = []
    year = start.year
    while True:
        for last_month in quarter_last_months:
            qdate = date(year, last_month, 1)
            if qdate > end:
                return pairs
            if qdate >= start:
                pred_month = last_month - 3 + flash_month
                pairs.append((qdate.isoformat(), _month_end(year, pred_month)))
        year += 1

######## Function 4: Fetch Historical Data ########
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

######## Function 5: Fetch Evaluation Metrics (RMSE) ########
## The following function fetches RMSE values for a single model ##
def fetch_rmse(client, models: list[str]) -> dict[str, dict]:

    result = client.table("rmse") \
        .select("model", "version", "rmse") \
        .in_("model", models) \
        .execute()

    # Accumulate rmse values per model across all versions
    totals: dict[str, list[float]] = {}
    for row in result.data:
        model = row["model"]
        if model not in totals:
            totals[model] = []
        totals[model].append(row["rmse"])

    # Return the mean rmse across versions for each model
    metrics = {}
    for model, values in totals.items():
        metrics[model] = {"rmse": sum(values) / len(values)}

    return metrics

######## Function 5: Fetch Evaluation Metrics (DM) ########
## Fetches pairwise DM test statistics for all unique pairs among `models` ##
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

def fetch_realised_gdp(client, quarter: str) -> float | None:
    quarter_start = quarter_to_dates(quarter)
    result = client.table("gdp") \
        .select("GDPC1_t") \
        .eq("sasdate", quarter_start) \
        .execute()
    if result.data:
        return result.data[0]["GDPC1_t"]
    return None

# Display name -> database name mapping
MODEL_DB_NAMES = {
    "Ensemble":       "All_Model_Average",
    "RF Lags Avg":    "RF_Lags_Average",
    "RF Lags UMIDAS": "RF_Lags_UMIDAS",
    "LASSO UMIDAS":   "LASSO_UMIDAS",
}
DEFAULT_MODELS = ["Ensemble"]
MODELS = list(MODEL_DB_NAMES.keys())

MODEL_COLORS = {
    "Ensemble": "#1f77b4",
    "RF Lags Avg": "#2ca02c",
    "RF Lags UMIDAS": "#d62728",
    "LASSO UMIDAS": "#ff7f0e",
}

MODEL_DESCRIPTIONS = {
    "Ensemble": "A simple average of 5 backend models: 3 LASSO variants (simple average, simple average + lags, U-MIDAS) and 2 Random Forest variants (simple average + lags, U-MIDAS + lags). Only 3 of these 5 models are shown individually in this app.",
    "RF Lags Avg": "A Random Forest Bridge Equation model using simple quarterly averages of monthly data. Includes lags of the quarterly averages as features.",
    "RF Lags UMIDAS": "A Random Forest using U-MIDAS to treat each month within the quarter as a separate input, not a quarterly average. Includes quarterly lags of both the monthly U-MIDAS block and the quarterly variables as features.",
    "LASSO UMIDAS": "A Regularized U-MIDAS regression using monthly variables from the current quarter only.",
}

def to_db_names(display_names: list[str]) -> list[str]:
    """Translate a list of display names to DB names for fetch functions."""
    return [MODEL_DB_NAMES[m] for m in display_names if m in MODEL_DB_NAMES]

def from_db_name(db_name: str) -> str:
    """Translate a single DB name back to its display name."""
    return {v: k for k, v in MODEL_DB_NAMES.items()}.get(db_name, db_name)

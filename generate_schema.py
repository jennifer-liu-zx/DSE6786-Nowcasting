"""
generate_schema.py
==================
Reads the 5 CSV files in data/ and auto-generates supabase/schema.sql.

Run this from the project root:
    python generate_schema.py

Re-run it whenever the model team changes the CSV column structure.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_FILE = Path("database/schema.sql")

CSV_FILES = {
    "gdp": DATA_DIR / "gdp.csv",
    "fred_md": DATA_DIR / "fred_md.csv",
    "fred_qd_x": DATA_DIR / "fred_qd_X.csv",
    "filled_md": DATA_DIR / "filled_md.csv",
    "filled_qd": DATA_DIR / "filled_qd.csv",
}

# Define the schema type for each column
def get_sql_type(column_name: str) -> str:
    if column_name == "sasdate":
        return "DATE NOT NULL"
    return "NUMERIC"

# Generating the CREATE TABLE statements
def generate_create_table(table_name:str, columns: list[str]) -> str:
    lines = []
    lines.append(f"CREATE TABLE IF NOT EXISTS {table_name} (")

    for col in columns:
        sql_type = get_sql_type(col)
        if col == "sasdate":
            lines.append(f"    {col} {sql_type},")
        else:
            lines.append(f'    "{col}" {sql_type},')

    lines[-1] = lines[-1].rstrip(",")  # Remove the last comma
    lines.append("    PRIMARY KEY (sasdate)")
    lines.append(");")
    return "\n".join(lines)

# HELPERS

def get_columns(filepath: Path) -> list[str]:
    df = pd.read_csv(filepath, nrows=0)  # Read only the header
    return df.columns.tolist()

def format_column(col: str) -> str:
    if col == "sasdate":
        return "sasdate"
    return f'"{col}"' # Wrap column names in double quotes to handle special characters

def gen_input_table(table_name: str, columns: list[str]) -> str:
    lines = []
    lines.append(f"--{'-'*60}")
    lines.append(f"-- TABLE: {table_name}")
    lines.append(f"--{'-'*60}")
    lines.append(f"")
    lines.append(f"CREATE TABLE IF NOT EXISTS {table_name} (")

    col_defs = []
    for col in columns:
        if col == "sasdate":
            col_defs.append(f"    {col} DATE NOT NULL")
        else:
            col_defs.append(f'    {format_column(col)} NUMERIC')
    lines.append(",\n".join(col_defs) + ",")

    lines.append("    PRIMARY KEY (sasdate)")
    lines.append(");")
    lines.append("")
    return "\n".join(lines)

def gen_rls(table_name: str) -> str:
    return "\n".join([
        f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY;",
        f"",
        f'DROP POLICY IF EXISTS "anon can read {table_name}" ON {table_name};',
        f'CREATE POLICY "anon can read {table_name}"',
        f"    ON {table_name}",
        f"    FOR SELECT",
        f"    TO anon",
        f"    USING (true);",
        f"",
    ])

def generate_model_forecasts_table() -> str:
    return "\n".join([
            f"-- {'─' * 60}",
            f"-- TABLE: model_forecasts",
            f"-- {'─' * 60}",
            f"",
            f"CREATE TABLE IF NOT EXISTS model_forecasts (",
            f"    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),",
            f"    run_date      DATE        NOT NULL,",
            f"    model_name    TEXT        NOT NULL,",
            f"    quarter_date       DATE        NOT NULL,",
            f"    month_date    DATE        NOT NULL,",
            f"    nowcast       NUMERIC     NOT NULL,",
            f"    ci_50_lb      NUMERIC     NOT NULL,",
            f"    ci_50_ub      NUMERIC     NOT NULL,",
            f"    ci_80_lb      NUMERIC     NOT NULL,",
            f"    ci_80_ub      NUMERIC     NOT NULL,",
            f"",
            f"    -- Upsert key: one forecast per model per month",
            f"    -- When pipeline revises a month's estimate, this",
            f"    -- constraint ensures we overwrite rather than duplicate",
            f"    CONSTRAINT model_forecasts_unique UNIQUE (model_name, quarter_date, month_date)",
            f");",
            f"",
            f"CREATE INDEX IF NOT EXISTS idx_model_forecasts_month_date",
            f"    ON model_forecasts (month_date);",
            f"",
            f"CREATE INDEX IF NOT EXISTS idx_model_forecasts_model_name",
            f"    ON model_forecasts (model_name);",
            f"",
        ])

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

# MAIN
def main():
    OUTPUT_FILE.parent.mkdir(exist_ok=True)  # Ensure the output directory exists

    blocks = []
    blocks.append("\n".join([
        "-- ================================================================",
        "-- schema.sql",
        "-- Auto-generated by generate_schema.py",
        "-- Do not edit manually. Re-run generate_schema.py to update.",
        "-- Run this file in the Supabase SQL editor to set up your database.",
        "-- ================================================================",
        "",
    ]))

    for table_name, filepath in CSV_FILES.items():
        print(f"Reading columns from {filepath} ...")
        columns = get_columns(filepath)
        print(f"  {len(columns)} columns found.")

        blocks.append(gen_input_table(table_name, columns))
        blocks.append(gen_rls(table_name))
    
    blocks.append(generate_model_forecasts_table())
    blocks.append(gen_rls("model_forecasts"))
    blocks.append(generate_dm_test_table())
    blocks.append(gen_rls("dm_test"))
    blocks.append(generate_evaluation_table())
    blocks.append(gen_rls("evaluation"))
    blocks.append(generate_rmse_table())
    blocks.append(gen_rls("rmse"))

    schema_sql = "\n".join(blocks)
    OUTPUT_FILE.write_text(schema_sql, encoding="utf-8")
    print(f"\nDone. Schema written to {OUTPUT_FILE}")
    print("Paste the contents of this file into the Supabase SQL editor to create your tables.")

if __name__ == "__main__":
    main()

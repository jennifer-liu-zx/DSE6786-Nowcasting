"""
run_pipeline_range.py
----------------------
Runs pipeline.pipe.run() once for every month-end date in a given range.
Useful for backfilling several months of nowcasts in one go.

Run with (from the project root):
    python -m scripts.run_pipeline_range --start 2026-01-01 --end 2026-03-31
"""

import argparse

import pandas as pd

from pipeline.pipe import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Range start date, e.g. 2026-01-01")
    parser.add_argument("--end", required=True, help="Range end date, e.g. 2026-03-31")
    args = parser.parse_args()

    month_ends = pd.date_range(start=args.start, end=args.end, freq="M")
    for date in month_ends:
        run(run_date=date.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    main()

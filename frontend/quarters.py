import calendar
from datetime import date


# To automate quarters based on today's date, we define a helper function that maps any date to its current and previous quarter in "YYYY:QX" format. This ensures our app always offers up-to-date quarter options without manual updates.
def date_to_quarter(system_date: date) -> dict:
    # First month of each quarter
    quarter_first_months = {1: 1, 2: 4, 3: 7, 4: 10}

    raw_quarter = (system_date.month - 1) // 3 + 1
    current_year = system_date.year

    # Last day of the first month of the current raw quarter
    first_month = quarter_first_months[raw_quarter]
    last_day_of_first_month = calendar.monthrange(current_year, first_month)[1]
    threshold = date(current_year, first_month, last_day_of_first_month)

    # If we haven't passed the end of the first month, stay in the previous quarter
    if system_date < threshold:
        raw_quarter -= 1
        if raw_quarter == 0:
            raw_quarter = 4
            current_year -= 1

    current_quarter = raw_quarter

    if current_quarter == 1:
        previous_quarter = 4
        previous_year = current_year - 1
    else:
        previous_quarter = current_quarter - 1
        previous_year = current_year

    return {
        "current_quarter": f"{current_year}:Q{current_quarter}",
        "previous_quarter": f"{previous_year}:Q{previous_quarter}",
    }


def shift_quarter(quarter_str: str, n: int) -> str:
    """Shift a 'YYYY:QX' string by n quarters (n can be negative)."""
    year_str, q_str = quarter_str.split(":Q")
    year, q = int(year_str), int(q_str)
    total = (year * 4 + (q - 1)) + n
    new_year, new_q = divmod(total, 4)
    return f"{new_year}:Q{new_q + 1}"


QUARTERS = [
    date_to_quarter(date.today())["current_quarter"],
    date_to_quarter(date.today())["previous_quarter"],
    shift_quarter(date_to_quarter(date.today())["previous_quarter"], -1),
]

# Split app.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the non-reactive, clearly-separable pieces of `app.py` (1143 lines) into a new `frontend/` package, shrinking `app.py` by ~460 lines without changing any behavior.

**Architecture:** A new `frontend/` package (empty `__init__.py`, matching the existing `pipeline/`/`database/` convention) gains four modules: `frontend/quarters.py` (date-to-quarter arithmetic), `frontend/model_labels.py` (model display-name/color/description metadata), `frontend/theme.py` (the `THEME` dict, plus a new `get_theme()` helper that removes a 7-way duplicated line), and `frontend/tour.py` (the onboarding-tour UI helpers and their copy strings). `app.py`'s `app_ui` and `server()` are otherwise untouched — this is an extract-only split; `server()`'s ~605 lines of reactive logic are explicitly *not* decomposed (a separate, not-yet-approved future candidate).

**Tech Stack:** Python 3.13, Shiny for Python, shinywidgets/Plotly.

## Global Constraints

- No pytest/unittest exists in this repo — verification is standalone `python3` scripts with `assert` statements, matching the existing `if __name__ == "__main__":` idiom.
- Work happens in an isolated git worktree off current `main` (create via `EnterWorktree`, name it `split-app-py`; the paths below assume `.claude/worktrees/split-app-py`). Every implementer must verify `pwd`/`git rev-parse HEAD`/`git branch --show-current` before doing anything, and again before committing. Re-copy `.env` into the worktree after the post-creation `git merge main --ff-only` — that merge step drops `.env` from the worktree's working directory since `main` no longer tracks it.
- `frontend/__init__.py` is empty — `pipeline/__init__.py` and `database/__init__.py` are both empty in this repo, and this plan follows that convention rather than turning `__init__.py` into a re-export surface.
- Naming: the new metadata module is `frontend/model_labels.py`, **not** `frontend/models.py` — `pipeline/models/` already exists and holds actual ML model implementations (`AR_benchmark.py`, `lasso.py`, `rf.py`); a same-named-but-different-purpose file would be confusing to find later.
- Renaming convention inside `frontend/tour.py`: drop the leading underscore only on names `app.py` actually imports and calls from outside the file (`centered_modal`, `spotlight`, `info_icon`, and the 10 copy-string constants). Keep the underscore on `_tooltip_base`, `_btn_row`, `_close_btn`, and `_BTN_MARGIN` — confirmed via `grep` that these four are called only from within `tour.py` itself (by `centered_modal`/`spotlight`/`_btn_row`), never from `app.py`.
- Out of scope, do not touch: decomposing `server()`'s reactive logic itself; the `TODO: swap get_dummy_metrics → fetch_evaluation_metrics` comment inside `eval_metrics()` (pre-existing, unrelated).
- Nothing outside `app.py` imports from it (confirmed via repo-wide `grep` before writing this plan), so no other file needs updating for any of these moves.
- Every code block below was hand-verified against the real, current `app.py` on `main` at commit `f33ebe6` before this plan was written — exact line numbers may drift by the time a task executes (earlier tasks in this plan edit the file), so match on the literal code shown, not on line numbers.

---

## Task 1: Create `frontend/` package and `frontend/quarters.py`

**Files:**
- Create: `frontend/__init__.py` (empty)
- Create: `frontend/quarters.py`
- Modify: `app.py` (imports section, currently lines 9–61)

**Interfaces:**
- Produces: `date_to_quarter(system_date: date) -> dict`, `shift_quarter(quarter_str: str, n: int) -> str`, `QUARTERS: list[str]` (a 3-element list, computed once at import time from `date.today()`) — all in `frontend.quarters`.

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py")

import ast

src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py/app.py").read()
ast.parse(src)
assert "from frontend.quarters import QUARTERS" in src
assert "def date_to_quarter" not in src, "date_to_quarter should have moved out of app.py entirely"
assert "def shift_quarter" not in src, "shift_quarter should have moved out of app.py entirely"

print("ALL TASK 1 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: from frontend.quarters import QUARTERS not found`

- [ ] **Step 3: Create `frontend/__init__.py`**

Empty file (matches `pipeline/__init__.py` and `database/__init__.py`, both empty in this repo).

- [ ] **Step 4: Create `frontend/quarters.py`**

```python
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
```

- [ ] **Step 5: Edit `app.py`**

Replace (currently lines 9–61 — the blank line after the `database.client` import, through the closing `]` of `QUARTERS`):

```python

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
```

with:

```python

from frontend.quarters import QUARTERS
```

(The whole block — `import calendar`, the duplicate `from datetime import date`, both functions, and the `QUARTERS` computation — is replaced by a single import line. The top-level `from datetime import date` at line 6 already covers every other `date(...)` usage remaining in `app.py`.)

- [ ] **Step 6: Run the verification script again**

Expected: `ALL TASK 1 CHECKS PASSED`

- [ ] **Step 7: Functional check — `QUARTERS` is still correct**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py")

from frontend.quarters import QUARTERS, date_to_quarter, shift_quarter
from datetime import date

assert len(QUARTERS) == 3
assert all(":Q" in q for q in QUARTERS), f"expected 'YYYY:QX' format, got {QUARTERS}"

# date_to_quarter is deterministic for a fixed date — spot-check a known case
result = date_to_quarter(date(2026, 5, 15))
assert result == {"current_quarter": "2026:Q2", "previous_quarter": "2026:Q1"}, result

assert shift_quarter("2026:Q1", -1) == "2025:Q4"
assert shift_quarter("2026:Q1", 1) == "2026:Q2"

print("QUARTERS:", QUARTERS)
print("FUNCTIONAL CHECK OK")
```

- [ ] **Step 8: Commit**

```bash
git add frontend/__init__.py frontend/quarters.py app.py
git commit -m "Extract quarter-date helpers from app.py into frontend/quarters.py"
```

---

## Task 2: Create `frontend/model_labels.py`

**Files:**
- Create: `frontend/model_labels.py`
- Modify: `app.py` (model-metadata block, currently right after Task 1's new import line)

**Interfaces:**
- Produces: `MODEL_DB_NAMES: dict[str, str]`, `DEFAULT_MODELS: list[str]`, `MODELS: list[str]`, `MODEL_COLORS: dict[str, str]`, `MODEL_DESCRIPTIONS: dict[str, str]`, `to_db_names(display_names: list[str]) -> list[str]`, `from_db_name(db_name: str) -> str` — all in `frontend.model_labels`.

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py")

import ast

src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py/app.py").read()
ast.parse(src)
assert "from frontend.model_labels import MODEL_DB_NAMES, DEFAULT_MODELS, MODELS, MODEL_COLORS, MODEL_DESCRIPTIONS, to_db_names, from_db_name" in src
assert "MODEL_DB_NAMES = {" not in src, "MODEL_DB_NAMES should have moved out of app.py entirely"
assert "def to_db_names" not in src, "to_db_names should have moved out of app.py entirely"

print("ALL TASK 2 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: from frontend.model_labels import ... not found`

- [ ] **Step 3: Create `frontend/model_labels.py`**

```python
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
```

- [ ] **Step 4: Edit `app.py`**

Replace (currently right after Task 1's `from frontend.quarters import QUARTERS` line):

```python
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
```

with:

```python
from frontend.model_labels import MODEL_DB_NAMES, DEFAULT_MODELS, MODELS, MODEL_COLORS, MODEL_DESCRIPTIONS, to_db_names, from_db_name
```

- [ ] **Step 5: Run the verification script again**

Expected: `ALL TASK 2 CHECKS PASSED`

- [ ] **Step 6: Functional check**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py")

from frontend.model_labels import MODEL_DB_NAMES, DEFAULT_MODELS, MODELS, MODEL_COLORS, MODEL_DESCRIPTIONS, to_db_names, from_db_name

assert to_db_names(["Ensemble"]) == ["All_Model_Average"]
assert to_db_names(["Ensemble", "LASSO UMIDAS"]) == ["All_Model_Average", "LASSO_UMIDAS"]
assert from_db_name("All_Model_Average") == "Ensemble"
assert from_db_name("not_a_real_model") == "not_a_real_model"  # falls back to the input unchanged
assert MODELS == ["Ensemble", "RF Lags Avg", "RF Lags UMIDAS", "LASSO UMIDAS"]
assert DEFAULT_MODELS == ["Ensemble"]
assert set(MODEL_COLORS) == set(MODELS)
assert set(MODEL_DESCRIPTIONS) == set(MODELS)

print("FUNCTIONAL CHECK OK")
```

- [ ] **Step 7: Commit**

```bash
git add frontend/model_labels.py app.py
git commit -m "Extract model display-metadata from app.py into frontend/model_labels.py"
```

---

## Task 3: Create `frontend/theme.py` with a `get_theme()` helper

**Files:**
- Create: `frontend/theme.py`
- Modify: `app.py` (the `THEME` dict block, and all 7 call sites of `t = THEME["dark"] if is_dark.get() else THEME["light"]`)

**Interfaces:**
- Produces: `THEME: dict` (keys `"light"`/`"dark"`, each a dict of color/font keys — unchanged structure), `get_theme(is_dark: bool) -> dict` — both in `frontend.theme`.

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py")

import ast

src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py/app.py").read()
ast.parse(src)
assert "from frontend.theme import get_theme" in src
assert "THEME = {" not in src, "THEME dict should have moved out of app.py entirely"
assert 'THEME["dark"] if is_dark.get() else THEME["light"]' not in src, "all 7 duplicated theme-selection lines must be replaced with get_theme(is_dark.get())"
assert src.count("t = get_theme(is_dark.get())") == 7, f"expected 7 call sites using get_theme(), found {src.count('t = get_theme(is_dark.get())')}"

print("ALL TASK 3 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: from frontend.theme import get_theme not found`

- [ ] **Step 3: Create `frontend/theme.py`**

```python
# =============================================================================
# THEME — edit colours and fonts here
# =============================================================================
#
# Each mode has the following keys:
#   bg_page         — page / outermost background
#   bg_card         — card body background
#   bg_card_header  — card header strip background
#   text_primary    — main body text
#   text_secondary  — muted / label text
#   accent          — buttons, active tabs, highlights (secondary accent colour)
#   border          — card / input borders
#   grid            — plot gridlines
#   plot_bg         — plot area background (passed directly to Plotly)
#   plot_paper      — plot paper background (passed directly to Plotly)
#   plot_text       — axis labels / tick text colour in Plotly
#
# FONTS
#   font_body       — applied to <body>; controls all UI text
#   font_heading    — applied to h1–h3
#
# To load a Google Font, add a ui.tags.link() in app_ui and reference it here,
# e.g. font_body = "'Inter', sans-serif"

THEME = {
    "light": {
        # ── Backgrounds ──────────────────────────────────────────────────────
        "bg_page":        "#ffffff",   # white page
        "bg_card":        "#f8f9fa",   # light grey card body
        "bg_card_header": "#f0f2f5",   # slightly deeper grey card header
        # ── Text ─────────────────────────────────────────────────────────────
        "text_primary":   "#1a2366",   # dark navy
        "text_secondary": "#6c757d",   # muted grey
        # ── Accent ───────────────────────────────────────────────────────────
        "accent":         "#1a2366",   # dark navy
        # ── Borders & grids ──────────────────────────────────────────────────
        "border":         "#dee2e6",   # standard grey border
        "grid":           "#e9ecef",   # light grey plot grid
        # ── Plotly surface colours ────────────────────────────────────────────
        "plot_bg":        "#ffffff",
        "plot_paper":     "#ffffff",
        "plot_text":      "#1a2366",
        # ── Model line colours ────────────────────────────────────────────────
        "model_colors": {
            "Ensemble":           "#005f9e",   # deep accessible blue
            "RF Lags Avg":        "#237523",   # dark green
            "RF Lags UMIDAS":        "#b83232",   # dark red
            "LASSO UMIDAS":        "#f7c948",   # bright yellow
        },
        # ── Button hover ─────────────────────────────────────────────────────
        "btn_hover":      "#e2e6ea",   # slightly darker than bg_card_header
        # ── Images ───────────────────────────────────────────────────────────
        "logo_src":       "blue_logo.png",
        "wordmark_src":   "blue_wordmark.png",
        # ── Fonts ─────────────────────────────────────────────────────────────
        # TODO: replace with your chosen font stack, e.g. "'Inter', sans-serif"
        "font_body":      "'Hanken Grotesk', sans-serif",
        "font_heading":   "'Hanken Grotesk', sans-serif",
    },
    "dark": {
        # ── Backgrounds ──────────────────────────────────────────────────────
        "bg_page":        "#1a1d21",
        "bg_card":        "#2b2f35",
        "bg_card_header": "#22262c",
        # ── Text ─────────────────────────────────────────────────────────────
        "text_primary":   "#e9ecef",
        "text_secondary": "#adb5bd",
        # ── Accent ───────────────────────────────────────────────────────────
        "accent":         "#4dabf7",
        # ── Borders & grids ──────────────────────────────────────────────────
        "border":         "#3d4249",
        "grid":           "#3d4249",
        # ── Plotly surface colours ────────────────────────────────────────────
        "plot_bg":        "#2b2f35",
        "plot_paper":     "#2b2f35",
        "plot_text":      "#e9ecef",
        # ── Model line colours ────────────────────────────────────────────────
        "model_colors": {
            "Ensemble": "#5bc0f8",
            "RF Lags Avg":        "#5dd55d",
            "RF Lags UMIDAS":        "#ff6b6b",
            "LASSO UMIDAS":        "#f7c948",
        },
        # ── Button hover ─────────────────────────────────────────────────────
        "btn_hover":      "#2e333a",   # slightly lighter than bg_card_header
        # ── Images ───────────────────────────────────────────────────────────
        "logo_src":       "white_logo.png",
        "wordmark_src":   "white_wordmark.png",
        # ── Fonts ─────────────────────────────────────────────────────────────
        # TODO: replace with your chosen font stack (can differ from light mode)
        "font_body":      "'Hanken Grotesk', sans-serif",
        "font_heading":   "'Hanken Grotesk', sans-serif",
    },
}

# =============================================================================
# END THEME
# =============================================================================


def get_theme(is_dark: bool) -> dict:
    """Return the active THEME sub-dict for the given dark-mode state."""
    return THEME["dark"] if is_dark else THEME["light"]
```

- [ ] **Step 4: Edit `app.py` — remove the `THEME` block, add the import**

Replace (currently right after Task 2's model-metadata import line):

```python

# =============================================================================
# THEME — edit colours and fonts here
# =============================================================================
#
# Each mode has the following keys:
#   bg_page         — page / outermost background
#   bg_card         — card body background
#   bg_card_header  — card header strip background
#   text_primary    — main body text
#   text_secondary  — muted / label text
#   accent          — buttons, active tabs, highlights (secondary accent colour)
#   border          — card / input borders
#   grid            — plot gridlines
#   plot_bg         — plot area background (passed directly to Plotly)
#   plot_paper      — plot paper background (passed directly to Plotly)
#   plot_text       — axis labels / tick text colour in Plotly
#
# FONTS
#   font_body       — applied to <body>; controls all UI text
#   font_heading    — applied to h1–h3
#
# To load a Google Font, add a ui.tags.link() in app_ui and reference it here,
# e.g. font_body = "'Inter', sans-serif"

THEME = {
    "light": {
        # ── Backgrounds ──────────────────────────────────────────────────────
        "bg_page":        "#ffffff",   # white page
        "bg_card":        "#f8f9fa",   # light grey card body
        "bg_card_header": "#f0f2f5",   # slightly deeper grey card header
        # ── Text ─────────────────────────────────────────────────────────────
        "text_primary":   "#1a2366",   # dark navy
        "text_secondary": "#6c757d",   # muted grey
        # ── Accent ───────────────────────────────────────────────────────────
        "accent":         "#1a2366",   # dark navy
        # ── Borders & grids ──────────────────────────────────────────────────
        "border":         "#dee2e6",   # standard grey border
        "grid":           "#e9ecef",   # light grey plot grid
        # ── Plotly surface colours ────────────────────────────────────────────
        "plot_bg":        "#ffffff",
        "plot_paper":     "#ffffff",
        "plot_text":      "#1a2366",
        # ── Model line colours ────────────────────────────────────────────────
        "model_colors": {
            "Ensemble":           "#005f9e",   # deep accessible blue
            "RF Lags Avg":        "#237523",   # dark green
            "RF Lags UMIDAS":        "#b83232",   # dark red
            "LASSO UMIDAS":        "#f7c948",   # bright yellow
        },
        # ── Button hover ─────────────────────────────────────────────────────
        "btn_hover":      "#e2e6ea",   # slightly darker than bg_card_header
        # ── Images ───────────────────────────────────────────────────────────
        "logo_src":       "blue_logo.png",
        "wordmark_src":   "blue_wordmark.png",
        # ── Fonts ─────────────────────────────────────────────────────────────
        # TODO: replace with your chosen font stack, e.g. "'Inter', sans-serif"
        "font_body":      "'Hanken Grotesk', sans-serif",
        "font_heading":   "'Hanken Grotesk', sans-serif",
    },
    "dark": {
        # ── Backgrounds ──────────────────────────────────────────────────────
        "bg_page":        "#1a1d21",
        "bg_card":        "#2b2f35",
        "bg_card_header": "#22262c",
        # ── Text ─────────────────────────────────────────────────────────────
        "text_primary":   "#e9ecef",
        "text_secondary": "#adb5bd",
        # ── Accent ───────────────────────────────────────────────────────────
        "accent":         "#4dabf7",
        # ── Borders & grids ──────────────────────────────────────────────────
        "border":         "#3d4249",
        "grid":           "#3d4249",
        # ── Plotly surface colours ────────────────────────────────────────────
        "plot_bg":        "#2b2f35",
        "plot_paper":     "#2b2f35",
        "plot_text":      "#e9ecef",
        # ── Model line colours ────────────────────────────────────────────────
        "model_colors": {
            "Ensemble": "#5bc0f8",
            "RF Lags Avg":        "#5dd55d",
            "RF Lags UMIDAS":        "#ff6b6b",
            "LASSO UMIDAS":        "#f7c948",
        },
        # ── Button hover ─────────────────────────────────────────────────────
        "btn_hover":      "#2e333a",   # slightly lighter than bg_card_header
        # ── Images ───────────────────────────────────────────────────────────
        "logo_src":       "white_logo.png",
        "wordmark_src":   "white_wordmark.png",
        # ── Fonts ─────────────────────────────────────────────────────────────
        # TODO: replace with your chosen font stack (can differ from light mode)
        "font_body":      "'Hanken Grotesk', sans-serif",
        "font_heading":   "'Hanken Grotesk', sans-serif",
    },
}

# =============================================================================
# END THEME
# =============================================================================
```

with:

```python

from frontend.theme import get_theme
```

- [ ] **Step 5: Edit `app.py` — replace all 7 duplicated theme-selection lines**

Each of the following 7 occurrences (verbatim, identical each time) inside `server()`:

```python
        t = THEME["dark"] if is_dark.get() else THEME["light"]
```

becomes:

```python
        t = get_theme(is_dark.get())
```

They occur inside these 7 functions — edit each independently, don't rely on find-and-replace-all tooling silently doing the wrong thing if indentation ever differs: `theme_css()`, `logo_img()`, `wizard_ui()`, `dm_overlay()`, `models_overlay()`, `nowcast_plot()`, `historical_plot()`.

- [ ] **Step 6: Run the verification script again**

Expected: `ALL TASK 3 CHECKS PASSED`

- [ ] **Step 7: Functional check**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py")

from frontend.theme import THEME, get_theme

assert get_theme(True) == THEME["dark"]
assert get_theme(False) == THEME["light"]
assert get_theme(True) is not get_theme(False)
assert set(THEME["light"].keys()) == set(THEME["dark"].keys())
assert "bg_page" in THEME["light"] and "font_body" in THEME["light"]

print("FUNCTIONAL CHECK OK")
```

- [ ] **Step 8: Commit**

```bash
git add frontend/theme.py app.py
git commit -m "Extract THEME dict into frontend/theme.py; add get_theme() to remove 7-way duplication"
```

---

## Task 4: Create `frontend/tour.py`

**Files:**
- Create: `frontend/tour.py`
- Modify: `app.py` (onboarding-tour helpers block, and every call site of the renamed functions/constants)

**Interfaces:**
- Produces (public — imported by `app.py`): `ABOUT_NOWCASTING`, `QUARTER_SELECTION`, `MODEL_SELECTION`, `CONFIDENCE_INTERVAL`, `CONFIDENCE_INTERVAL_HOVER`, `RMSE_HOVER`, `HISTORICAL_DATA`, `DATE_RANGE_SELECTION`, `FLASH_ESTIMATE`, `EVALUATION_METRICS` (all `str`), `centered_modal(header: str, body: str | None, step: int, t: dict, show_logo: bool = False)`, `spotlight(selector: str, tooltip_pos: str, description: str, step: int, t: dict)`, `info_icon(tooltip_text)` — all in `frontend.tour`.
- Internal only (not imported by `app.py`): `_tooltip_base(t: dict) -> str`, `_btn_row(step: int)`, `_close_btn(t: dict)`, `_BTN_MARGIN`.

- [ ] **Step 1: Write the failing verification script**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py")

import ast

src = open("/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py/app.py").read()
ast.parse(src)
assert "from frontend.tour import" in src
assert "_ABOUT_NOWCASTING" not in src, "underscore-prefixed copy constants should be gone from app.py"
assert "def _centered_modal" not in src and "def _spotlight" not in src and "def _info_icon" not in src, \
    "tour helper functions should have moved out of app.py entirely"
assert "_centered_modal(" not in src and "_spotlight(" not in src and "_info_icon(" not in src, \
    "app.py's call sites must use the renamed (non-underscore) names"
assert src.count("centered_modal(") >= 2
assert src.count("spotlight(") >= 8
assert src.count("info_icon(") == 3

print("ALL TASK 4 CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `AssertionError: from frontend.tour import not found`

- [ ] **Step 3: Create `frontend/tour.py`**

```python
from shiny import ui

##Text edit
ABOUT_NOWCASTING = "Traditional GDP data is released with a significant delay, leaving policymakers and businesses flying blind for months. Nowcasting solves this by using higher-frequency indicators—like retail sales and industrial output—to provide a real-time estimate of economic growth. By bridging this gap, we can identify turning points in the business cycle in real time, rather than after a delay."
QUARTER_SELECTION = "Toggle between quarters to view their evolving GDP Nowcast. Each data point on the timeline represents a new prediction for that quarter's growth, updated monthly as fresh economic indicators provide information that wasn't available in previous months."
MODEL_SELECTION = "Select one or more models to compare different econometric approaches simultaneously. By default, the Ensemble Model is displayed, providing a balanced view by aggregating inputs from multiple sub-models."
CONFIDENCE_INTERVAL = "Visualise the uncertainty of a model by toggling its Confidence Intervals. To maintain clarity on the chart, you can view the probability bands for only one model at a time."
CONFIDENCE_INTERVAL_HOVER = "Range of values within which the true GDP growth rate is likely to fall, with a certain probability."
RMSE_HOVER = "Root Mean Square Error, measures average prediction error in percentage points; Smaller RMSE indicates better performance."
HISTORICAL_DATA = "Curious about how the model performs historically?"
DATE_RANGE_SELECTION = "Select a specific timeline to evaluate how our model’s historical Nowcasts tracked against official Realised GDP. This view helps to visualise the model’s accuracy and bias during past economic cycles or periods of high volatility."
FLASH_ESTIMATE = "Since our model predicts a single quarter’s GDP multiple times as new data arrives, the Flash Estimate setting lets you choose from predictions available at different points during the quarter."
EVALUATION_METRICS = "Compare the statistical performance of your selected models."

_BTN_MARGIN = "margin-right: 8px;"


def _tooltip_base(t: dict) -> str:
    return (
        f"position: fixed; background: {t['bg_card']}; color: {t['text_primary']}; "
        "padding: 1.2rem 1.5rem; border-radius: 8px; z-index: 1001; "
        f"min-width: 240px; max-width: 320px; border: 1px solid {t['border']}; "
        "box-shadow: 0 4px 20px rgba(0,0,0,0.4);"
    )


def _btn_row(step: int):
    buttons = []
    if step >= 2:
        buttons.append(
            ui.input_action_button("wizard_prev", "←", style=_BTN_MARGIN)
        )
    if step == 1:
        buttons.append(
            ui.input_action_button("wizard_skip", "Skip tutorial", style=_BTN_MARGIN)
        )
        buttons.append(ui.input_action_button("wizard_next", "Show me around"))
    elif step == 6:
        pass  # no Next — user must click the Historical Data tab to advance
    elif step == 10:
        buttons.append(ui.input_action_button("wizard_finish", "Finish tutorial"))
    else:
        buttons.append(ui.input_action_button("wizard_next", "→"))
    return ui.div(*buttons, style="margin-top: 1rem;")


def info_icon(tooltip_text):
    """Inline ⓘ icon that shows a floating tooltip on hover."""
    return ui.span(
        ui.tags.span("ⓘ", class_="tt-icon"),
        ui.tags.span(tooltip_text, class_="tt-box"),
        class_="tt-wrap",
    )


def _close_btn(t: dict):
    return ui.input_action_button(
        "wizard_close", "×",
        style=(
            "position: absolute; top: 0.75rem; right: 0.75rem; "
            "background: none; border: none; font-size: 1.25rem; "
            f"color: {t['text_secondary']}; cursor: pointer; padding: 0; line-height: 1;"
        ),
    )


def centered_modal(header: str, body: str | None, step: int, t: dict, show_logo: bool = False):
    content = []
    if show_logo:
        content.append(ui.img(src=t["wordmark_src"], style="height: 48px; display: block; margin-bottom: 1rem;"))
    content.append(ui.h3(header, style="margin-bottom: 1rem;"))
    if body:
        content.append(ui.p(body))
    content.append(_btn_row(step))
    return ui.div(
        _close_btn(t),
        *content,
        style=(
            "position: fixed; top: 50%; left: 50%; "
            "transform: translate(-50%, -50%); "
            f"background: {t['bg_card']}; color: {t['text_primary']}; "
            f"border: 1px solid {t['border']}; "
            "padding: 2.5rem; border-radius: 10px; "
            "min-width: 360px; max-width: 540px; "
            "box-shadow: 0 0 0 9999px rgba(0,0,0,0.7), "
            "0 4px 30px rgba(0,0,0,0.4); "
            "z-index: 1001; pointer-events: auto; position: fixed;"
        ),
    )


def spotlight(selector: str, tooltip_pos: str, description: str, step: int, t: dict):
    """
    Spotlight overlay: the target element gets a massive box-shadow that
    darkens everything else. A floating tooltip sits next to it.
    """
    css = f"""
        {selector} {{
            position: relative !important;
            z-index: 1000 !important;
            box-shadow: 0 0 0 9999px rgba(0,0,0,0.7) !important;
            border-radius: 6px;
        }}
    """
    hint = ""
    if step == 6:
        hint = ui.p(
            ui.tags.em("Click the 'Historical Data' tab to continue."),
            style=f"margin-top: 0.5rem; font-size: 0.85rem; color: {t['text_secondary']};",
        )
    return ui.div(
        ui.tags.style(css),
        ui.div(
            _close_btn(t),
            ui.p(description, style="margin-bottom: 0.25rem;"),
            hint,
            _btn_row(step),
            style=f"{_tooltip_base(t)} {tooltip_pos} position: fixed;",
        ),
    )
```

- [ ] **Step 4: Edit `app.py` — remove the onboarding-tour helpers block, add the import**

Replace (currently right after Task 3's `from frontend.theme import get_theme` line):

```python

# ── Onboarding wizard helpers ─────────────────────────────────────────────────

##Text edit
_ABOUT_NOWCASTING = "Traditional GDP data is released with a significant delay, leaving policymakers and businesses flying blind for months. Nowcasting solves this by using higher-frequency indicators—like retail sales and industrial output—to provide a real-time estimate of economic growth. By bridging this gap, we can identify turning points in the business cycle in real time, rather than after a delay."
_QUARTER_SELECTION = "Toggle between quarters to view their evolving GDP Nowcast. Each data point on the timeline represents a new prediction for that quarter's growth, updated monthly as fresh economic indicators provide information that wasn't available in previous months."
_MODEL_SELECTION = "Select one or more models to compare different econometric approaches simultaneously. By default, the Ensemble Model is displayed, providing a balanced view by aggregating inputs from multiple sub-models."
_CONFIDENCE_INTERVAL = "Visualise the uncertainty of a model by toggling its Confidence Intervals. To maintain clarity on the chart, you can view the probability bands for only one model at a time."
_CONFIDENCE_INTERVAL_HOVER = "Range of values within which the true GDP growth rate is likely to fall, with a certain probability."
_RMSE_HOVER = "Root Mean Square Error, measures average prediction error in percentage points; Smaller RMSE indicates better performance."
_HISTORICAL_DATA = "Curious about how the model performs historically?"
_DATE_RANGE_SELECTION = "Select a specific timeline to evaluate how our model’s historical Nowcasts tracked against official Realised GDP. This view helps to visualise the model’s accuracy and bias during past economic cycles or periods of high volatility."
_FLASH_ESTIMATE = "Since our model predicts a single quarter’s GDP multiple times as new data arrives, the Flash Estimate setting lets you choose from predictions available at different points during the quarter."
_EVALUATION_METRICS = "Compare the statistical performance of your selected models."

_BTN_MARGIN = "margin-right: 8px;"


def _tooltip_base(t: dict) -> str:
    return (
        f"position: fixed; background: {t['bg_card']}; color: {t['text_primary']}; "
        "padding: 1.2rem 1.5rem; border-radius: 8px; z-index: 1001; "
        f"min-width: 240px; max-width: 320px; border: 1px solid {t['border']}; "
        "box-shadow: 0 4px 20px rgba(0,0,0,0.4);"
    )


def _btn_row(step: int):
    buttons = []
    if step >= 2:
        buttons.append(
            ui.input_action_button("wizard_prev", "←", style=_BTN_MARGIN)
        )
    if step == 1:
        buttons.append(
            ui.input_action_button("wizard_skip", "Skip tutorial", style=_BTN_MARGIN)
        )
        buttons.append(ui.input_action_button("wizard_next", "Show me around"))
    elif step == 6:
        pass  # no Next — user must click the Historical Data tab to advance
    elif step == 10:
        buttons.append(ui.input_action_button("wizard_finish", "Finish tutorial"))
    else:
        buttons.append(ui.input_action_button("wizard_next", "→"))
    return ui.div(*buttons, style="margin-top: 1rem;")


def _info_icon(tooltip_text):
    """Inline ⓘ icon that shows a floating tooltip on hover."""
    return ui.span(
        ui.tags.span("ⓘ", class_="tt-icon"),
        ui.tags.span(tooltip_text, class_="tt-box"),
        class_="tt-wrap",
    )


def _close_btn(t: dict):
    return ui.input_action_button(
        "wizard_close", "×",
        style=(
            "position: absolute; top: 0.75rem; right: 0.75rem; "
            "background: none; border: none; font-size: 1.25rem; "
            f"color: {t['text_secondary']}; cursor: pointer; padding: 0; line-height: 1;"
        ),
    )


def _centered_modal(header: str, body: str | None, step: int, t: dict, show_logo: bool = False):
    content = []
    if show_logo:
        content.append(ui.img(src=t["wordmark_src"], style="height: 48px; display: block; margin-bottom: 1rem;"))
    content.append(ui.h3(header, style="margin-bottom: 1rem;"))
    if body:
        content.append(ui.p(body))
    content.append(_btn_row(step))
    return ui.div(
        _close_btn(t),
        *content,
        style=(
            "position: fixed; top: 50%; left: 50%; "
            "transform: translate(-50%, -50%); "
            f"background: {t['bg_card']}; color: {t['text_primary']}; "
            f"border: 1px solid {t['border']}; "
            "padding: 2.5rem; border-radius: 10px; "
            "min-width: 360px; max-width: 540px; "
            "box-shadow: 0 0 0 9999px rgba(0,0,0,0.7), "
            "0 4px 30px rgba(0,0,0,0.4); "
            "z-index: 1001; pointer-events: auto; position: fixed;"
        ),
    )


def _spotlight(selector: str, tooltip_pos: str, description: str, step: int, t: dict):
    """
    Spotlight overlay: the target element gets a massive box-shadow that
    darkens everything else. A floating tooltip sits next to it.
    """
    css = f"""
        {selector} {{
            position: relative !important;
            z-index: 1000 !important;
            box-shadow: 0 0 0 9999px rgba(0,0,0,0.7) !important;
            border-radius: 6px;
        }}
    """
    hint = ""
    if step == 6:
        hint = ui.p(
            ui.tags.em("Click the 'Historical Data' tab to continue."),
            style=f"margin-top: 0.5rem; font-size: 0.85rem; color: {t['text_secondary']};",
        )
    return ui.div(
        ui.tags.style(css),
        ui.div(
            _close_btn(t),
            ui.p(description, style="margin-bottom: 0.25rem;"),
            hint,
            _btn_row(step),
            style=f"{_tooltip_base(t)} {tooltip_pos} position: fixed;",
        ),
    )
```

with:

```python

from frontend.tour import (
    ABOUT_NOWCASTING, QUARTER_SELECTION, MODEL_SELECTION, CONFIDENCE_INTERVAL,
    CONFIDENCE_INTERVAL_HOVER, RMSE_HOVER, HISTORICAL_DATA, DATE_RANGE_SELECTION,
    FLASH_ESTIMATE, EVALUATION_METRICS, centered_modal, spotlight, info_icon,
)
```

- [ ] **Step 5: Edit `app.py` — update call sites in `nowcast_controls`/`historical_controls`**

Replace:
```python
        ui.card_header(ui.span("Confidence Interval"), _info_icon(_CONFIDENCE_INTERVAL_HOVER)),
```
with:
```python
        ui.card_header(ui.span("Confidence Interval"), info_icon(CONFIDENCE_INTERVAL_HOVER)),
```

Replace:
```python
        ui.div(ui.strong("FLASH ESTIMATE USED"), _info_icon(_FLASH_ESTIMATE), style="display:inline-flex;align-items:center;"),
```
with:
```python
        ui.div(ui.strong("FLASH ESTIMATE USED"), info_icon(FLASH_ESTIMATE), style="display:inline-flex;align-items:center;"),
```

- [ ] **Step 6: Edit `app.py` — update `wizard_ui()`'s call sites**

Replace (inside `wizard_ui()`):

```python
        if step == 1:
            return _centered_modal("US GDP Nowcast", None, step, t, show_logo=True)
        if step == 2:
            return _centered_modal(
                "About Nowcasting", _ABOUT_NOWCASTING, step, t)
        if step == 3:
            return _spotlight(
                "#card_quarter",
                "right: 36%; top: 20%;",
                _QUARTER_SELECTION, step, t,
            )
        if step == 4:
            return _spotlight(
                "#card_nowcast_model",
                "right: 36%; top: 37%;",
                _MODEL_SELECTION, step, t,
            )
        if step == 5:
            return _spotlight(
                "#card_ci",
                "right: 36%; top: 56%;",
                _CONFIDENCE_INTERVAL, step, t,
            )
        if step == 6:
            return _spotlight(
                ".nav-tabs li:nth-child(2) .nav-link",
                "left: 35%; top: 7%;",
                _HISTORICAL_DATA, step, t,
            )
        if step == 7:
            return _spotlight(
                "#card_date_range",
                "right: 36%; top: 22%;",
                _DATE_RANGE_SELECTION, step, t,
            )
        if step == 8:
            return _spotlight(
                "#card_hist_display",
                "right: 36%; top: 38%;",
                _MODEL_SELECTION, step, t,
            )
        if step == 9:
            return _spotlight(
                "#card_hist_display",
                "right: 36%; top: 66%;",
                _FLASH_ESTIMATE, step, t,
            )
        if step == 10:
            return _spotlight(
                "#card_eval",
                "right: 36%; top: 68%;",
                _EVALUATION_METRICS, step, t,
            )
```

with:

```python
        if step == 1:
            return centered_modal("US GDP Nowcast", None, step, t, show_logo=True)
        if step == 2:
            return centered_modal(
                "About Nowcasting", ABOUT_NOWCASTING, step, t)
        if step == 3:
            return spotlight(
                "#card_quarter",
                "right: 36%; top: 20%;",
                QUARTER_SELECTION, step, t,
            )
        if step == 4:
            return spotlight(
                "#card_nowcast_model",
                "right: 36%; top: 37%;",
                MODEL_SELECTION, step, t,
            )
        if step == 5:
            return spotlight(
                "#card_ci",
                "right: 36%; top: 56%;",
                CONFIDENCE_INTERVAL, step, t,
            )
        if step == 6:
            return spotlight(
                ".nav-tabs li:nth-child(2) .nav-link",
                "left: 35%; top: 7%;",
                HISTORICAL_DATA, step, t,
            )
        if step == 7:
            return spotlight(
                "#card_date_range",
                "right: 36%; top: 22%;",
                DATE_RANGE_SELECTION, step, t,
            )
        if step == 8:
            return spotlight(
                "#card_hist_display",
                "right: 36%; top: 38%;",
                MODEL_SELECTION, step, t,
            )
        if step == 9:
            return spotlight(
                "#card_hist_display",
                "right: 36%; top: 66%;",
                FLASH_ESTIMATE, step, t,
            )
        if step == 10:
            return spotlight(
                "#card_eval",
                "right: 36%; top: 68%;",
                EVALUATION_METRICS, step, t,
            )
```

- [ ] **Step 7: Edit `app.py` — update `eval_metrics()`'s call site**

Replace:
```python
            _info_icon(_RMSE_HOVER),
```
with:
```python
            info_icon(RMSE_HOVER),
```

- [ ] **Step 8: Run the verification script again**

Expected: `ALL TASK 4 CHECKS PASSED`

- [ ] **Step 9: Functional check — the moved UI helpers still build valid Shiny UI**

```python
import sys
sys.path.insert(0, "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py")

from frontend.tour import centered_modal, spotlight, info_icon, ABOUT_NOWCASTING, RMSE_HOVER
from frontend.theme import get_theme

t = get_theme(False)

# These raise if the internal (still-underscored) helpers they call —
# _tooltip_base, _btn_row, _close_btn — aren't wired up correctly.
modal = centered_modal("Test", "body text", 2, t)
assert modal is not None

spot = spotlight("#some-selector", "left: 10%;", "description text", 3, t)
assert spot is not None

icon = info_icon("hover text")
assert icon is not None

assert isinstance(ABOUT_NOWCASTING, str) and len(ABOUT_NOWCASTING) > 0
assert isinstance(RMSE_HOVER, str) and len(RMSE_HOVER) > 0

print("FUNCTIONAL CHECK OK")
```

- [ ] **Step 10: Commit**

```bash
git add frontend/tour.py app.py
git commit -m "Extract onboarding-tour UI helpers into frontend/tour.py, renaming boundary-crossing names"
```

---

## Task 5: Full-repo consistency sweep

**Files:**
- None modified — verification only.

**Interfaces:**
- None — this task confirms Tasks 1–4 compose correctly and `app.py` still works end-to-end.

- [ ] **Step 1: Confirm `app.py`'s size dropped roughly as expected**

```python
worktree = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py"
with open(f"{worktree}/app.py") as f:
    line_count = sum(1 for _ in f)
print(f"app.py is now {line_count} lines")
assert line_count < 750, f"expected app.py to have shrunk to well under 750 lines (started at 1143), got {line_count}"
```

- [ ] **Step 2: Confirm no stray old names remain anywhere in `app.py`**

```python
worktree = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py"
src = open(f"{worktree}/app.py").read()

stale_names = [
    "THEME[", "MODEL_DB_NAMES = {", "def date_to_quarter", "def shift_quarter",
    "def to_db_names", "def from_db_name", "_ABOUT_NOWCASTING", "_QUARTER_SELECTION",
    "_MODEL_SELECTION", "_CONFIDENCE_INTERVAL", "_RMSE_HOVER", "_HISTORICAL_DATA",
    "_DATE_RANGE_SELECTION", "_FLASH_ESTIMATE", "_EVALUATION_METRICS",
    "_centered_modal(", "_spotlight(", "_info_icon(",
]
found = [name for name in stale_names if name in src]
assert not found, f"stale references still in app.py: {found}"

import ast
ast.parse(src)
print("NO STALE REFERENCES — app.py parses cleanly")
```

- [ ] **Step 3: Confirm `app.py` imports cleanly without Supabase credentials**

Mirrors the check from the prior Supabase-client-seam plan, confirming this split didn't reintroduce an import-time dependency on live credentials:

```python
import subprocess, os
stripped_env = {k: v for k, v in os.environ.items() if not k.startswith("SUPABASE")}
worktree = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py"
result = subprocess.run(
    ["python3", "-c", "import app; print('APP IMPORTED WITH NO SUPABASE ENV VARS')"],
    cwd=worktree,
    capture_output=True, text=True, env=stripped_env,
)
print(result.stdout[-1500:])
print(result.stderr[-1500:])
assert result.returncode == 0, f"app.py should still import cleanly with no Supabase credentials: {result.stderr}"
assert "APP IMPORTED WITH NO SUPABASE ENV VARS" in result.stdout
print("APP IMPORT-WITHOUT-CREDENTIALS CHECK OK")
```

- [ ] **Step 4: Confirm the Shiny `App` object itself still constructs**

`app.py`'s final line builds `app = App(app_ui, server, static_assets=...)` at import time — this exercises the full `app_ui` tree (which now pulls in `QUARTERS`, `MODELS`, `DEFAULT_MODELS`, and the `info_icon`/copy-string call sites from Task 4) without needing a live Shiny session:

```python
import subprocess, os
stripped_env = {k: v for k, v in os.environ.items() if not k.startswith("SUPABASE")}
worktree = "/Users/Jennifur/Desktop/random-projects/DSE6786-Nowcasting/.claude/worktrees/split-app-py"
result = subprocess.run(
    ["python3", "-c", "from app import app; print('APP OBJECT TYPE:', type(app).__name__)"],
    cwd=worktree,
    capture_output=True, text=True, env=stripped_env,
)
print(result.stdout[-1500:])
print(result.stderr[-1500:])
assert result.returncode == 0, f"app object should construct cleanly: {result.stderr}"
assert "APP OBJECT TYPE: App" in result.stdout
print("APP OBJECT CONSTRUCTION CHECK OK")
```

- [ ] **Step 5: Report the clean sweep**

No commit needed for this task if the sweep passes — it's verification-only. If any step fails, fix the underlying issue in the relevant earlier task's files and re-run this whole task before considering the plan complete.

---

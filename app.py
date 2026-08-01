from pathlib import Path
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import numpy as np
from datetime import date
from pipeline.fetch_functions import fetch_nowcast_data, fetch_confidence_intervals, fetch_historical_data, fetch_rmse, fetch_dm, fetch_realised_gdp

import calendar
from datetime import date
from dateutil.relativedelta import relativedelta

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

QUARTERS = [
    date_to_quarter(date.today())["current_quarter"],
    date_to_quarter(date.today())["previous_quarter"],
    date_to_quarter(date.today() - relativedelta(months=3))["previous_quarter"],
]

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
            ui.input_action_button("wizard_prev", "\u2190", style=_BTN_MARGIN)
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
        buttons.append(ui.input_action_button("wizard_next", "\u2192"))
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


# ── UI ────────────────────────────────────────────────────────────────────────

nowcast_controls = ui.div(
    ui.card(
        ui.card_header("Quarter Selection"),
        ui.input_radio_buttons(
            "quarter",
            None,
            choices=QUARTERS,
            selected=QUARTERS[0],
            inline=True,
        ),
        id="card_quarter",
    ),
    ui.card(
        ui.card_header("Model Selection"),
        ui.input_checkbox_group(
            "nowcast_models",
            None,
            choices=MODELS,
            selected=DEFAULT_MODELS,
        ),
        ui.input_action_button(
            "view_nowcast_models", "About our models",
            style="width: 100%; margin-top: 0.25rem;",
        ),
        id="card_nowcast_model",
    ),
    ui.card(
        ui.card_header(ui.span("Confidence Interval"), _info_icon(_CONFIDENCE_INTERVAL_HOVER)),
        ui.input_select(
            "ci_model",
            None,
            choices={"None": "None"},
            selected="None",
        ),
        id="card_ci",
    ),
)

historical_controls = ui.div(
    ui.card(
        ui.card_header("Date Range Selection"),
        ui.input_date_range(
            "hist_date_range",
            None,
            start="2022-01-01",
            end="2026-03-01",
        ),
        id="card_date_range",
    ),
    ui.card(
        ui.card_header("Display Options"),
        ui.div(ui.strong("MODEL SELECTION")),
        ui.input_checkbox_group(
            "hist_models",
            None,
            choices=MODELS,
            selected=DEFAULT_MODELS,
        ),
        ui.input_action_button(
            "view_hist_models", "About our models",
            style="width: 100%; margin-top: 0.25rem;",
        ),
        ui.div(ui.strong("FLASH ESTIMATE USED"), _info_icon(_FLASH_ESTIMATE), style="display:inline-flex;align-items:center;"),
        ui.input_select(
            "flash_month",
            None,
            choices={"1": "1st month", "2": "2nd month", "3": "3rd month"},
            selected="1",
        ),
        id="card_hist_display",
    ),
    ui.card(
        ui.card_header(ui.span("Evaluation Metrics")),
        ui.output_ui("eval_metrics"),
        id="card_eval",
    ),
)

_TOOLTIP_CSS = """
.card, .card-body, .card-header,
.shiny-input-container, .form-group {
    overflow: visible !important;
}
.tt-wrap {
    position: relative;
    display: inline-flex;
    align-items: center;
    margin-left: 6px;
    vertical-align: middle;
}
.tt-icon {
    cursor: default;
    font-size: 0.8rem;
    color: #6c757d;
    line-height: 1;
    user-select: none;
}
.tt-box {
    position: absolute;
    left: calc(100% + 8px);
    top: 50%;
    transform: translateY(-50%);
    background: #2b2f35;
    color: #e9ecef;
    padding: 0.55rem 0.75rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: normal;
    min-width: 180px;
    max-width: 260px;
    white-space: normal;
    line-height: 1.45;
    pointer-events: none;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.18s ease, visibility 0.18s ease;
    z-index: 1200;
    box-shadow: 0 3px 12px rgba(0,0,0,0.25);
}
.tt-wrap:hover .tt-box {
    opacity: 1;
    visibility: visible;
}
"""

_LOADING_CSS = """
#loading-screen {
    position: fixed;
    inset: 0;
    background: #ffffff;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: opacity 0.4s ease;
}
#loading-screen.fade-out {
    opacity: 0;
    pointer-events: none;
}
.loading-spinner {
    width: 36px;
    height: 36px;
    border: 3px solid #dee2e6;
    border-top-color: #1a2366;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-top: 1.5rem;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
"""

_LOADING_JS = """
$(document).one('shiny:idle', function() {
    var el = document.getElementById('loading-screen');
    if (el) {
        el.classList.add('fade-out');
        setTimeout(function() { el.remove(); }, 1500);
    }
});
"""

app_ui = ui.page_fluid(
    ui.tags.link(
        rel="stylesheet",
        href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:ital,wght@0,100..900;1,100..900&display=swap",
    ),
    ui.tags.style(_LOADING_CSS),
    ui.tags.style(_TOOLTIP_CSS),
    ui.div(
        ui.img(src="blue_logo.png", style="width: 72px;"),
        ui.div(class_="loading-spinner"),
        id="loading-screen",
    ),
    ui.tags.script(_LOADING_JS),
    ui.output_ui("theme_css"),
    ui.output_ui("wizard_ui"),
    ui.output_ui("dm_overlay"),
    ui.output_ui("models_overlay"),
    ui.div(
        ui.output_ui("logo_img"),
        ui.h1("US GDP Nowcast", style="margin: 0;"),
        ui.div(
            ui.output_ui("dark_mode_btn"),
            ui.input_action_button("wizard_replay", "Play tutorial", style="margin-left: 1rem;"),
            style="margin-left: auto; display: flex; align-items: center;",
        ),
        style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;",
    ),
    ui.navset_tab(
        ui.nav_panel(
            "Nowcast",
            ui.layout_columns(
                ui.card(output_widget("nowcast_plot")),
                nowcast_controls,
                col_widths=[8, 4],
            ),
        ),
        ui.nav_panel(
            "Historical Data",
            ui.layout_columns(
                ui.card(output_widget("historical_plot")),
                historical_controls,
                col_widths=[8, 4],
            ),
        ),
        id="main_tabs",
        selected="Nowcast",
    ),
    style="padding: 2rem 3rem 0 3rem;",
)


# ── Server ────────────────────────────────────────────────────────────────────

def server(input, output, session):

    wizard_step = reactive.value(1)
    dm_overlay_visible = reactive.value(False)
    models_overlay_visible = reactive.value(False)
    is_dark = reactive.value(False)

    # ── Theme CSS injection ───────────────────────────────────────────────────

    @render.ui
    def theme_css():
        t = THEME["dark"] if is_dark.get() else THEME["light"]
        css = f"""
            body {{
                background-color: {t['bg_page']} !important;
                color: {t['text_primary']} !important;
                font-family: {t['font_body']} !important;  /* ← body font */
            }}
            h1, h2, h3, h4, h5, h6 {{
                font-family: {t['font_heading']} !important;  /* ← heading font */
                font-weight: bold !important;
                color: {t['text_primary']} !important;
            }}
            .card {{
                background-color: {t['bg_card']} !important;
                border-color: {t['border']} !important;
                color: {t['text_primary']} !important;
            }}
            .card-header {{
                background-color: {t['bg_card_header']} !important;
                border-color: {t['border']} !important;
                color: {t['text_primary']} !important;
            }}
            .nav-tabs {{
                border-color: {t['border']} !important;
            }}
            .nav-tabs .nav-link {{
                color: {t['text_secondary']} !important;
            }}
            .nav-tabs .nav-link.active {{
                background-color: {t['bg_card']} !important;
                border-color: {t['border']} !important;
                color: {t['accent']} !important;
            }}
            label, .form-label, .shiny-input-container {{
                color: {t['text_primary']} !important;
            }}
            .form-control, .form-select {{
                background-color: {t['bg_card']} !important;
                border-color: {t['border']} !important;
                color: {t['text_primary']} !important;
            }}
            .btn-default, .btn-secondary {{
                background-color: {t['bg_card_header']} !important;
                border-color: {t['border']} !important;
                color: {t['text_primary']} !important;
                transition: background-color 0.15s ease, border-color 0.15s ease;
            }}
            .btn-default:hover, .btn-secondary:hover {{
                background-color: {t['btn_hover']} !important;
                border-color: {t['border']} !important;
            }}
            /* ── Secondary accent: active/focus highlights ── */
            .btn-primary, a {{
                color: {t['accent']} !important;
            }}
            .btn-primary:hover {{
                background-color: {t['btn_hover']} !important;
                border-color: {t['border']} !important;
                transition: background-color 0.15s ease;
            }}
            /* ── Tab content padding ── */
            .tab-content > .tab-pane {{
                padding-top: 1.25rem;
            }}
            /* ── Tighter checkbox spacing in model selection ── */
            #card_nowcast_model .form-check,
            #card_hist_display .form-check {{
                margin-bottom: 0.1rem;
            }}
            #card_nowcast_model .shiny-input-container,
            #card_hist_display .shiny-input-container {{
                margin-bottom: 0;
            }}
        """
        return ui.tags.style(css)

    @render.ui
    def dark_mode_btn():
        label = "View in light mode" if is_dark.get() else "View in dark mode"
        return ui.input_action_button("toggle_dark_mode", label)

    @render.ui
    def logo_img():
        t = THEME["dark"] if is_dark.get() else THEME["light"]
        return ui.img(src=t["logo_src"], style="width: 60px;")
    
    @render.ui
    def wordmark_img():
        return ui.img(src="blue_wordmark.png", style="width: 60px;")

    @reactive.effect
    @reactive.event(input.toggle_dark_mode)
    def _on_toggle_dark():
        is_dark.set(not is_dark.get())

    # ── Wizard rendering ──────────────────────────────────────────────────────

    @render.ui
    def wizard_ui():
        step = wizard_step.get()
        if step == 0:
            return ui.div()
        t = THEME["dark"] if is_dark.get() else THEME["light"]
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
        return ui.div()

    # ── Wizard navigation ─────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.wizard_next)
    def _on_next():
        new = wizard_step.get() + 1
        if new == 7:
            ui.update_navs("main_tabs", selected="Historical Data")
        elif new <= 6:
            ui.update_navs("main_tabs", selected="Nowcast")
        wizard_step.set(new)

    @reactive.effect
    @reactive.event(input.wizard_prev)
    def _on_prev():
        new = wizard_step.get() - 1
        if new <= 6:
            ui.update_navs("main_tabs", selected="Nowcast")
        else:
            ui.update_navs("main_tabs", selected="Historical Data")
        wizard_step.set(new)

    @reactive.effect
    @reactive.event(input.wizard_skip)
    def _on_skip():
        wizard_step.set(0)

    @reactive.effect
    @reactive.event(input.wizard_close)
    def _on_close():
        wizard_step.set(0)

    @reactive.effect
    @reactive.event(input.wizard_finish)
    def _on_finish():
        wizard_step.set(0)

    @reactive.effect
    @reactive.event(input.wizard_replay)
    def _on_replay():
        ui.update_navs("main_tabs", selected="Nowcast")
        wizard_step.set(1)

    # Advance from step 6 when the user clicks the Historical Data tab
    @reactive.effect
    @reactive.event(input.main_tabs)
    def _tab_advance():
        if wizard_step.get() == 6 and input.main_tabs() == "Historical Data":
            wizard_step.set(7)

    # ── DM overlay show/hide ──────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.view_dm_stats)
    def _show_dm_overlay():
        dm_overlay_visible.set(True)

    @reactive.effect
    @reactive.event(input.close_dm_overlay)
    def _hide_dm_overlay():
        dm_overlay_visible.set(False)

    @render.ui
    def dm_overlay():
        if not dm_overlay_visible.get():
            return ui.div()

        selected_models = list(input.hist_models() or DEFAULT_MODELS)
        flash_month = int(input.flash_month() or "1")
        t = THEME["dark"] if is_dark.get() else THEME["light"]
        db_models = to_db_names(selected_models)

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

        # RMSE column
        _rmse_label = ui.div(
            ui.tags.u(ui.strong("RMSE"))
        )
        rmse_lines = [_rmse_label]
        for model in selected_models:
            db_name = MODEL_DB_NAMES.get(model)
            if db_name is not None and db_name in metrics:
                rmse_lines.append(ui.p(f"{model}: {metrics[db_name]['rmse']:.1f}"))

        return ui.div(
            # Backdrop
            ui.div(style=(
                "position: fixed; inset: 0; background: rgba(0,0,0,0.6); "
                "z-index: 1100;"
            )),
            # Panel
            ui.div(
                # Header row
                ui.div(
                    ui.h3("Diebold-Mariano Test Statistics", style="margin: 0;"),
                    ui.input_action_button(
                        "close_dm_overlay", "×",
                        style=(
                            "background: none; border: none; font-size: 1.5rem; "
                            f"color: {t['text_primary']}; cursor: pointer; "
                            "padding: 0; line-height: 1;"
                        ),
                    ),
                    style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;",
                ),
                ui.p(
                    "The Diebold-Mariano test checks whether one model's forecast errors are significantly smaller than another's. A test statistic further from zero than ±1.96 (5%) or ±1.64 (10%) indicates a significant difference; the sign shows which model is favored.",
                    style=f"color: {t['text_secondary']}; margin-bottom: 1.25rem;",
                ),
                # Two columns
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
                style=(
                    f"position: fixed; top: 50%; left: 50%; "
                    "transform: translate(-50%, -50%); "
                    f"background: {t['bg_card']}; color: {t['text_primary']}; "
                    "padding: 2rem; border-radius: 10px; "
                    "z-index: 1101; pointer-events: auto; "
                    "min-width: 480px; max-width: 85vw; max-height: 85vh; overflow-y: auto; "
                    f"box-shadow: 0 4px 30px rgba(0,0,0,0.4);"
                ),
            ),
        )
    # ── Models overlay show/hide ──────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.view_nowcast_models, input.view_hist_models)
    def _show_models_overlay():
        models_overlay_visible.set(True)

    @reactive.effect
    @reactive.event(input.close_models_overlay)
    def _hide_models_overlay():
        models_overlay_visible.set(False)

    @render.ui
    def models_overlay():
        if not models_overlay_visible.get():
            return ui.div()

        t = THEME["dark"] if is_dark.get() else THEME["light"]

        # Build model details content
        model_cards = []
        for model in MODELS:
            model_cards.append(
                ui.div(
                    ui.p(ui.strong(model), style="margin: 0 0 0.35rem 0; font-size: 0.95rem;"),
                    ui.p(
                        MODEL_DESCRIPTIONS.get(model, ""),
                        style=f"color: {t['text_secondary']}; margin-bottom: 0;",
                    ),
                    style=(
                        f"padding: 1rem; border: 1px solid {t['border']}; "
                        f"border-radius: 6px; margin-bottom: 0.75rem;"
                    ),
                )
            )

        return ui.div(
            # Backdrop
            ui.div(style=(
                "position: fixed; inset: 0; background: rgba(0,0,0,0.6); "
                "z-index: 1100;"
            )),
            # Panel
            ui.div(
                # Header row
                ui.div(
                    ui.h3("Model Descriptions", style="margin: 0;"),
                    ui.input_action_button(
                        "close_models_overlay", "×",
                        style=(
                            "background: none; border: none; font-size: 1.5rem; "
                            f"color: {t['text_secondary']}; cursor: pointer; "
                            "padding: 0; line-height: 1;"
                        ),
                    ),
                    style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;",
                ),
                ui.p(
                    "Review detailed information about our models.",
                    style=f"color: {t['text_secondary']}; margin-bottom: 1.25rem;",
                ),
                # Model cards
                ui.div(
                    *model_cards,
                    style="max-height: 60vh; overflow-y: auto;",
                ),
                style=(
                    f"position: fixed; top: 50%; left: 50%; "
                    "transform: translate(-50%, -50%); "
                    f"background: {t['bg_card']}; color: {t['text_primary']}; "
                    "padding: 2rem; border-radius: 10px; "
                    "z-index: 1101; pointer-events: auto; "
                    "min-width: 400px; max-width: 85vw; max-height: 85vh; overflow-y: auto; "
                    f"box-shadow: 0 4px 30px rgba(0,0,0,0.4);"
                ),
            ),
        )
    # ── Keep CI dropdown in sync with selected models ─────────────────────────

    @reactive.effect
    def _sync_ci_choices():
        selected = input.nowcast_models()
        choices = {"None": "None"}
        for m in (selected or []):
            choices[m] = m
        ui.update_select("ci_model", choices=choices, selected="None")

    # ── Nowcast plot ──────────────────────────────────────────────────────────

    @render_widget
    def nowcast_plot():
        quarter = input.quarter()
        selected_models = list(input.nowcast_models() or DEFAULT_MODELS)
        ci_model = input.ci_model()
        t = THEME["dark"] if is_dark.get() else THEME["light"]

        # TODO: swap get_dummy_nowcast_data → fetch_nowcast_data when Supabase ready
        data, x_labels = fetch_nowcast_data(quarter)

        fig = go.Figure()

        for model in selected_models:
            db_name = MODEL_DB_NAMES.get(model)
            if db_name is not None and db_name in data:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=data[db_name],
                        mode="lines+markers",
                        name=model,
                        line=dict(color=t["model_colors"].get(model, "#888"), width=2),
                    )
                )

        # Shaded confidence intervals (50% and 80%)
        if ci_model and ci_model != "None" and ci_model in selected_models:
            db_ci_model = MODEL_DB_NAMES.get(ci_model)
            if db_ci_model is not None:
                x_ci, ci50_lo, ci50_hi, ci80_lo, ci80_hi = fetch_confidence_intervals(quarter, db_ci_model)
            ci_color = MODEL_COLORS.get(ci_model, "#888")
            r, g, b = int(ci_color[1:3], 16), int(ci_color[3:5], 16), int(ci_color[5:7], 16)
            # 80% band (wider, more transparent) — drawn first so 50% renders on top
            fig.add_trace(
                go.Scatter(
                    x=x_ci + x_ci[::-1],
                    y=ci80_hi + ci80_lo[::-1],
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{ci_model} 80% CI",
                    showlegend=True,
                )
            )
            # 50% band (narrower, more opaque) — drawn on top
            fig.add_trace(
                go.Scatter(
                    x=x_ci + x_ci[::-1],
                    y=ci50_hi + ci50_lo[::-1],
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.25)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{ci_model} 50% CI",
                    showlegend=True,
                )
            )
        realised = fetch_realised_gdp(quarter)
        if realised is not None:
            fig.add_hline(
                y=realised,
                line_dash="dash",
                line_color=t["text_secondary"],
                line_width=1.5,
                annotation_text=f"Realised GDP: {realised:.1f}%",
                annotation_position="top left",
                annotation_font_color=t["text_secondary"],
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_labels[-1]],
                    y=[realised],
                    mode="markers",
                    name="Realised GDP",
                    marker=dict(
                        color=t["text_secondary"],
                        size=12,
                        symbol="diamond",
                        line=dict(color=t["plot_text"], width=1.5),
                    ),
                    hovertemplate="Realised GDP: %{y:.2f}%<extra></extra>",
                )
            )
        fig.update_layout(
            yaxis_title="% annual GDP growth",
            plot_bgcolor=t["plot_bg"],
            paper_bgcolor=t["plot_paper"],
            font=dict(color=t["plot_text"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=50, r=20, t=60, b=40),
            xaxis=dict(showgrid=True, gridcolor=t["grid"]),
            yaxis=dict(showgrid=True, gridcolor=t["grid"])
        )
        return fig

    # ── Historical plot ───────────────────────────────────────────────────────

    @render_widget
    def historical_plot():
        date_range = input.hist_date_range()
        start_date = date_range[0] if date_range else date(2020, 1, 1)
        end_date   = date_range[1] if date_range else date(2022, 1, 1)
        selected_models = list(input.hist_models() or DEFAULT_MODELS)
        flash_month = int(input.flash_month())
        t = THEME["dark"] if is_dark.get() else THEME["light"]

        # TODO: swap get_dummy_historical_data → fetch_historical_data when Supabase ready
        quarters, actual, predictions = fetch_historical_data(
                    start_date, end_date, flash_month
                )

        fig = go.Figure()

        # Actual GDP — dotted line
        actual_line_color = t["text_primary"]
        fig.add_trace(
            go.Scatter(
                x=quarters,
                y=actual,
                mode="lines+markers",
                name="Actual",
                line=dict(color=actual_line_color, width=2, dash="dot"),
            )
        )

        # Model predictions — solid lines
        for model in selected_models:
            db_name = MODEL_DB_NAMES.get(model)
            if db_name is not None and db_name in predictions:
                fig.add_trace(
                    go.Scatter(
                        x=quarters,
                        y=predictions[db_name],
                        mode="lines+markers",
                        name=model,
                        line=dict(color=t["model_colors"].get(model, "#888"), width=2),
                    )
                )

        fig.update_layout(
            yaxis_title="% annual GDP growth",
            plot_bgcolor=t["plot_bg"],
            paper_bgcolor=t["plot_paper"],
            font=dict(color=t["plot_text"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=50, r=20, t=60, b=40),
            xaxis=dict(showgrid=True, gridcolor=t["grid"]),
            yaxis=dict(showgrid=True, gridcolor=t["grid"])
        )
        return fig

    # ── Evaluation metrics ────────────────────────────────────────────────────

    @render.ui
    def eval_metrics():
        selected_models = list(input.hist_models() or DEFAULT_MODELS)
        if not selected_models:
            return ui.p("No models selected.")

        # TODO: swap get_dummy_metrics → fetch_evaluation_metrics when Supabase ready
        db_models = to_db_names(selected_models)
        metrics = fetch_rmse(db_models)

        rmse_lines = []
        for model in selected_models:
            db_name = MODEL_DB_NAMES.get(model)
            if db_name not in metrics:
                continue
            m = metrics[db_name]
            rmse_lines.append(ui.div(f"{model}: {m['rmse']:.1f}"))

        if not rmse_lines:
            return ui.p("No metrics available.")

        _rmse_label = ui.div(
            ui.tags.u(ui.strong("RMSE")),
            _info_icon(_RMSE_HOVER),
            style="display: inline-flex; align-items: center;",
        )
        content = [_rmse_label, *rmse_lines]
        if len(selected_models) > 1:
            content.append(
                ui.input_action_button(
                    "view_dm_stats", "View Diebold-Mariano test statistics",
                    style="margin-top: 0.75rem; width: 100%;",
                )
            )
        return ui.div(*content)


app = App(app_ui, server, static_assets=Path(__file__).parent / "www")


"""Onboarding-tour copy and UI helpers (modals, spotlights, info icons)."""

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

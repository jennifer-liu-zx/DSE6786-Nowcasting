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


def get_theme(is_dark: bool) -> dict:
    """Return the active THEME sub-dict for the given dark-mode state."""
    return THEME["dark"] if is_dark else THEME["light"]

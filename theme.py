"""
theme.py — Shared Theme & CSS for EduPredict System
====================================================
Provides a unified design system matching the Stitch mockup.
Uses Google's Outfit font (closest open-source to Google Sans).
Supports light and dark modes with a premium, differentiated look.
"""

import streamlit as st


def init_theme():
    """Initialize theme in session state. Default is light mode."""
    if "theme" not in st.session_state:
        st.session_state.theme = "light"


def toggle_theme():
    """Switch between light and dark mode."""
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"


def render_theme_toggle():
    """Render a styled theme toggle in the sidebar."""
    init_theme()
    is_dark = st.session_state.theme == "dark"
    label = "☀️  Switch to Light" if is_dark else "🌙  Switch to Dark"
    st.sidebar.button(label, on_click=toggle_theme, use_container_width=True,
                      key="theme_toggle_btn")


def get_theme_colors():
    """Return the current theme's full color dictionary."""
    init_theme()
    if st.session_state.theme == "dark":
        return {
            "bg":            "#0A0A0A",
            "bg_secondary":  "#141414",
            "surface":       "#1A1A1A",
            "surface_hover": "#242424",
            "primary":       "#4285F4",       # Google Blue
            "primary_hover": "#5A95F5",
            "text":          "#E8EAED",       # Google light text
            "text_secondary":"#9AA0A6",       # Google secondary gray
            "border":        "#303134",       # Google dark border
            "border_hover":  "#5F6368",
            "sidebar_bg":    "#111111",
            "sidebar_border":"#303134",
            "input_bg":      "#303134",
            "input_border":  "#5F6368",
            "btn_secondary_bg": "#303134",
            "btn_secondary_text": "#E8EAED",
            "btn_secondary_border": "#5F6368",
            "shadow":        "0 1px 3px rgba(0,0,0,0.5)",
            "shadow_hover":  "0 4px 16px rgba(0,0,0,0.6)",
            "excelling":     "#4285F4",
            "on_track":      "#34A853",       # Google Green
            "moderate_risk": "#FBBC04",       # Google Yellow
            "high_risk":     "#EA4335",       # Google Red
            "dynamic_risk":  "#A142F4",       # Google Purple
            "chart_template":"plotly_dark",
            "chart_bg":      "rgba(0,0,0,0)",
            "chart_paper":   "rgba(0,0,0,0)",
            "chart_grid":    "#303134",
        }
    else:
        return {
            "bg":            "#F8F9FA",       # Google background
            "bg_secondary":  "#F1F3F4",
            "surface":       "#FFFFFF",
            "surface_hover": "#F8F9FA",
            "primary":       "#1A73E8",       # Google Blue
            "primary_hover": "#1765CC",
            "text":          "#202124",       # Google dark text
            "text_secondary":"#5F6368",       # Google secondary
            "border":        "#DADCE0",       # Google border
            "border_hover":  "#BDC1C6",
            "sidebar_bg":    "#FFFFFF",
            "sidebar_border":"#DADCE0",
            "input_bg":      "#F1F3F4",
            "input_border":  "#DADCE0",
            "btn_secondary_bg": "#FFFFFF",
            "btn_secondary_text": "#1A73E8",
            "btn_secondary_border": "#DADCE0",
            "shadow":        "0 1px 3px rgba(60,64,67,0.15)",
            "shadow_hover":  "0 4px 16px rgba(60,64,67,0.2)",
            "excelling":     "#1A73E8",
            "on_track":      "#1E8E3E",       # Google Green
            "moderate_risk": "#F9AB00",       # Google Yellow
            "high_risk":     "#D93025",       # Google Red
            "dynamic_risk":  "#A142F4",       # Google Purple
            "chart_template":"plotly_white",
            "chart_bg":      "rgba(0,0,0,0)",
            "chart_paper":   "rgba(0,0,0,0)",
            "chart_grid":    "#DADCE0",
        }


def get_risk_color(risk_level, colors=None):
    """Get the color for a specific risk level."""
    if colors is None:
        colors = get_theme_colors()
    mapping = {
        "Excelling":     colors["excelling"],
        "On Track":      colors["on_track"],
        "Moderate Risk": colors["moderate_risk"],
        "High Risk":     colors["high_risk"],
        "Dynamic Risk":  colors["dynamic_risk"],
    }
    return mapping.get(risk_level, colors["text_secondary"])


def inject_css():
    """Inject the full EduPredict design system CSS."""
    c = get_theme_colors()
    is_dark = st.session_state.get("theme", "light") == "dark"
    
    # Dynamic shadow/glow adjustments
    card_shadow = "0 1px 2px rgba(0,0,0,0.3)" if is_dark else "0 1px 3px rgba(60,64,67,0.1)"
    card_hover_shadow = "0 4px 12px rgba(0,0,0,0.5)" if is_dark else "0 2px 8px rgba(60,64,67,0.15)"
    btn_primary_shadow = "0 2px 8px rgba(66,133,244,0.5)" if is_dark else "0 2px 8px rgba(26,115,232,0.3)"
    
    st.markdown(f"""
<style>
    /* ═══════════════════════════════════════════════════════
       EduPredict — Design System v2
       Google-inspired, premium interactive elements
       ═══════════════════════════════════════════════════════ */

    /* ── Fonts (Google's Outfit + Inter) ──────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
    }}

    .stApp {{
        background-color: {c['bg']};
    }}

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {c['sidebar_bg']} !important;
        border-right: 1px solid {c['sidebar_border']} !important;
    }}
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div {{
        color: {c['text']};
    }}
    [data-testid="stSidebar"] .stMarkdown p {{
        color: {c['text_secondary']} !important;
    }}

    /* ── Hide default Streamlit chrome ────────────────────── */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ── Typography ───────────────────────────────────────── */
    h1, h2, h3 {{
        font-family: 'Outfit', 'Inter', sans-serif !important;
        color: {c['text']} !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
    }}
    h1 {{ font-size: 2.2rem !important; line-height: 1.15 !important; }}
    h2 {{ font-size: 1.6rem !important; }}
    h3 {{ font-size: 1.2rem !important; }}

    p, li, span, label, div {{
        color: {c['text']};
    }}
    strong {{ font-weight: 700; }}

    /* ── EduPredict Custom Components ─────────────────────── */

    /* Page Header */
    .ep-page-header {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        color: {c['text']};
        margin-bottom: 2px;
        letter-spacing: -0.03em;
    }}
    .ep-page-sub {{
        color: {c['text_secondary']};
        font-size: 0.92rem;
        font-weight: 400;
        margin-bottom: 28px;
        line-height: 1.5;
    }}

    /* Metric Cards — elevated, Google Material style */
    .ep-metric {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 16px;
        padding: 22px 18px;
        text-align: center;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: {card_shadow};
    }}
    .ep-metric:hover {{
        border-color: {c['border_hover']};
        box-shadow: {card_hover_shadow};
        transform: translateY(-1px);
    }}
    .ep-metric-value {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .ep-metric-label {{
        color: {c['text_secondary']};
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        margin-top: 8px;
    }}

    /* Cards — elevated surfaces */
    .ep-card {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 16px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: {card_shadow};
    }}
    .ep-card:hover {{
        box-shadow: {card_hover_shadow};
    }}
    .ep-card-title {{
        font-family: 'Outfit', sans-serif;
        color: {c['text']};
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 14px;
        letter-spacing: -0.01em;
    }}
    .ep-card-desc {{
        color: {c['text_secondary']};
        font-size: 0.88rem;
        line-height: 1.6;
    }}

    /* Nav Cards */
    .ep-nav-card {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 16px;
        padding: 28px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: {card_shadow};
        height: 100%;
        cursor: pointer;
    }}
    .ep-nav-card:hover {{
        border-color: {c['primary']};
        box-shadow: 0 6px 24px rgba({'66,133,244' if is_dark else '26,115,232'},0.15);
        transform: translateY(-3px);
    }}
    .ep-nav-icon {{
        font-size: 2rem;
        margin-bottom: 14px;
        display: inline-block;
        background: {c['bg_secondary']};
        border-radius: 14px;
        width: 56px;
        height: 56px;
        line-height: 56px;
        text-align: center;
    }}
    .ep-nav-title {{
        font-family: 'Outfit', sans-serif;
        color: {c['text']};
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 8px;
    }}
    .ep-nav-desc {{
        color: {c['text_secondary']};
        font-size: 0.84rem;
        line-height: 1.55;
    }}

    /* Risk Badge (pill) */
    .ep-badge {{
        display: inline-block;
        padding: 7px 18px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }}

    /* Status Badge */
    .ep-status {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 16px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
    }}
    .ep-status-green {{
        background: {'rgba(52,168,83,0.15)' if is_dark else 'rgba(30,142,62,0.08)'};
        color: {c['on_track']};
        border: 1.5px solid {'rgba(52,168,83,0.3)' if is_dark else 'rgba(30,142,62,0.2)'};
    }}
    .ep-status-red {{
        background: {'rgba(234,67,53,0.15)' if is_dark else 'rgba(217,48,37,0.08)'};
        color: {c['high_risk']};
        border: 1.5px solid {'rgba(234,67,53,0.3)' if is_dark else 'rgba(217,48,37,0.2)'};
    }}

    /* Hero Section */
    .ep-hero {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 20px;
        padding: 52px 44px;
        margin-bottom: 32px;
        box-shadow: {card_shadow};
        position: relative;
    }}
    .ep-hero-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        color: {c['primary']};
        margin-bottom: 14px;
        letter-spacing: -0.04em;
    }}
    .ep-hero-desc {{
        color: {c['text_secondary']};
        font-size: 1.05rem;
        font-weight: 400;
        max-width: 640px;
        line-height: 1.7;
    }}

    /* Upload drop zone */
    .ep-dropzone {{
        background: {c['surface']};
        border: 2px dashed {c['border']};
        border-radius: 16px;
        padding: 52px;
        text-align: center;
        transition: all 0.25s ease;
        margin: 20px 0;
        box-shadow: {card_shadow};
    }}
    .ep-dropzone:hover {{
        border-color: {c['primary']};
        background: {c['bg_secondary']};
    }}

    /* Motivation box */
    .ep-motivation {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-left: 4px solid {c['primary']};
        border-radius: 16px;
        padding: 32px;
        margin: 28px 0;
        text-align: center;
        box-shadow: {card_shadow};
    }}
    .ep-motivation-emoji {{ font-size: 2.5rem; margin-bottom: 12px; }}
    .ep-motivation-text {{ color: {c['text_secondary']}; font-size: 1rem; line-height: 1.7; }}

    /* Info card */
    .ep-info-card {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 16px;
        box-shadow: {card_shadow};
    }}
    .ep-info-label {{
        color: {c['text_secondary']};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        font-weight: 600;
        margin-bottom: 8px;
    }}

    /* Sidebar brand */
    .ep-brand {{
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.2rem;
        color: {c['primary']} !important;
        letter-spacing: -0.03em;
    }}
    .ep-brand-sub {{
        font-size: 0.72rem;
        color: {c['text_secondary']} !important;
        margin-top: 2px;
        line-height: 1.4;
    }}

    /* Footer */
    .ep-footer {{
        text-align: center;
        color: {c['text_secondary']};
        font-size: 0.78rem;
        margin-top: 52px;
        padding: 24px 20px;
        border-top: 1px solid {c['border']};
    }}
    .ep-footer a {{ color: {c['primary']}; text-decoration: none; }}
    .ep-footer a:hover {{ text-decoration: underline; }}

    /* ═══════════════════════════════════════════════════════
       STREAMLIT COMPONENT OVERRIDES
       Makes buttons, inputs, toggles, etc. look premium
       ═══════════════════════════════════════════════════════ */

    /* ── All Buttons — elevated, Google Material 3 style ─── */
    .stButton>button {{
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        padding: 10px 24px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.01em !important;
        border: 1.5px solid {c['btn_secondary_border']} !important;
        background: {c['btn_secondary_bg']} !important;
        color: {c['btn_secondary_text']} !important;
        box-shadow: {card_shadow} !important;
    }}
    .stButton>button:hover {{
        box-shadow: {card_hover_shadow} !important;
        border-color: {c['primary']} !important;
        transform: translateY(-1px) !important;
        background: {c['surface_hover']} !important;
    }}
    .stButton>button:active {{
        transform: translateY(0) !important;
        box-shadow: {card_shadow} !important;
    }}
    /* Primary buttons — filled Google blue */
    .stButton>button[kind="primary"],
    .stButton>button[data-testid="stBaseButton-primary"] {{
        background: {c['primary']} !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: {btn_primary_shadow} !important;
    }}
    .stButton>button[kind="primary"]:hover,
    .stButton>button[data-testid="stBaseButton-primary"]:hover {{
        background: {c['primary_hover']} !important;
        box-shadow: 0 4px 16px rgba({'66,133,244' if is_dark else '26,115,232'},0.4) !important;
    }}

    /* ── Text Inputs — clear borders, focus glow ─────────── */
    .stTextInput>div>div>input {{
        border-radius: 12px !important;
        border: 1.5px solid {c['input_border']} !important;
        background: {c['input_bg']} !important;
        color: {c['text']} !important;
        padding: 12px 16px !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05) !important;
    }}
    .stTextInput>div>div>input:focus {{
        border-color: {c['primary']} !important;
        box-shadow: 0 0 0 3px rgba({'66,133,244' if is_dark else '26,115,232'},0.15) !important;
        outline: none !important;
    }}
    .stTextInput>div>div>input::placeholder {{
        color: {c['text_secondary']} !important;
        opacity: 0.7 !important;
    }}

    /* ── Selectbox — elevated dropdown ───────────────────── */
    .stSelectbox>div>div {{
        border-radius: 12px !important;
        border: 1.5px solid {c['input_border']} !important;
        background: {c['input_bg']} !important;
        transition: all 0.2s ease !important;
        box-shadow: {card_shadow} !important;
    }}
    .stSelectbox>div>div:hover {{
        border-color: {c['primary']} !important;
    }}

    /* ── Checkbox — visible square, blue check ───────────── */
    .stCheckbox > label > div[data-testid="stCheckbox"] {{
        border: 2px solid {c['input_border']} !important;
        border-radius: 6px !important;
        background: {c['input_bg']} !important;
        transition: all 0.15s ease !important;
    }}
    .stCheckbox > label > div[data-testid="stCheckbox"]:hover {{
        border-color: {c['primary']} !important;
    }}
    .stCheckbox label span {{
        color: {c['text']} !important;
        font-weight: 500 !important;
    }}

    /* ── File Uploader — styled drop area ────────────────── */
    [data-testid="stFileUploader"] {{
        border-radius: 14px !important;
    }}
    [data-testid="stFileUploader"] section {{
        border: 2px dashed {c['border']} !important;
        border-radius: 14px !important;
        padding: 24px !important;
        background: {c['surface']} !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: {c['primary']} !important;
        background: {c['bg_secondary']} !important;
    }}
    [data-testid="stFileUploader"] button {{
        background: {c['primary']} !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
    }}

    /* ── Expander — clean bordered container ─────────────── */
    div[data-testid="stExpander"] {{
        border: 1px solid {c['border']} !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: {card_shadow} !important;
        background: {c['surface']} !important;
    }}
    div[data-testid="stExpander"] summary {{
        font-weight: 600 !important;
        color: {c['text']} !important;
    }}

    /* ── Metric — Material-style card ────────────────────── */
    div[data-testid="stMetric"] {{
        background: {c['surface']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 14px !important;
        padding: 18px !important;
        box-shadow: {card_shadow} !important;
    }}
    div[data-testid="stMetric"] label {{
        color: {c['text_secondary']} !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {c['text']} !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }}

    /* ── Data Frame — bordered table ─────────────────────── */
    .stDataFrame {{
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 1px solid {c['border']} !important;
        box-shadow: {card_shadow} !important;
    }}

    /* ── Progress bar — Google blue ──────────────────────── */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {c['primary']}, {c['primary_hover']}) !important;
        border-radius: 8px !important;
    }}

    /* ── Download button — outlined style ────────────────── */
    .stDownloadButton>button {{
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: 1.5px solid {c['primary']} !important;
        color: {c['primary']} !important;
        background: transparent !important;
        box-shadow: none !important;
    }}
    .stDownloadButton>button:hover {{
        background: rgba({'66,133,244' if is_dark else '26,115,232'},0.08) !important;
        box-shadow: {card_shadow} !important;
    }}

    /* ── Alerts — softer look ────────────────────────────── */
    .stAlert {{
        border-radius: 14px !important;
        border: 1px solid {c['border']} !important;
    }}

    /* ── Tabs — clean underline ──────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        color: {c['text_secondary']};
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {c['primary']};
    }}

    /* ── Horizontal Rule ─────────────────────────────────── */
    hr {{
        border: none;
        border-top: 1px solid {c['border']};
        margin: 28px 0;
    }}
</style>
""", unsafe_allow_html=True)

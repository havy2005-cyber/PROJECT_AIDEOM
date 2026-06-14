import streamlit as st

NEON_BLUE = "#00D4FF"
NEON_PURPLE = "#BF00FF"
NEON_GREEN = "#00FF88"
NEON_ORANGE = "#FF6B35"
BG_DARK = "#0A0E27"
CARD_BG = "rgba(15, 20, 50, 0.75)"
SIDEBAR_BG = "linear-gradient(180deg, #0A0E27 0%, #1A1A3E 100%)"
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#E0E0FF", "family": "Arial"},
        "xaxis": {"gridcolor": "#1F2A5A", "linecolor": "#2A3A8A"},
        "yaxis": {"gridcolor": "#1F2A5A", "linecolor": "#2A3A8A"},
        "colorway": [NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, "#FFD700"],
    }
}

def apply_page_config():
    st.set_page_config(
        page_title="AIDEOM-VN | He thong ho tro phan tich kinh te Viet Nam",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def apply_global_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

    * { box-sizing: border-box; }

    .stApp {
        background: linear-gradient(135deg, #0A0E27 0%, #111830 40%, #0D1B2A 100%);
        background-attachment: fixed;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080D1F 0%, #101830 50%, #0A1020 100%) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.15);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #C0D0FF;
    }

    /* Main content padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Glassmorphism card */
    .glass-card {
        background: linear-gradient(135deg, rgba(15, 20, 50, 0.75) 0%, rgba(20, 30, 70, 0.6) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        margin-bottom: 1rem;
    }

    /* KPI card */
    .kpi-card {
        background: linear-gradient(135deg, rgba(10, 15, 40, 0.85) 0%, rgba(15, 25, 60, 0.7) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 212, 255, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5), 0 0 30px rgba(0, 212, 255, 0.1);
        border-color: rgba(0, 212, 255, 0.4);
    }
    .kpi-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 900;
        color: #00D4FF;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
        line-height: 1.1;
    }
    .kpi-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        color: #8090C0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
    }
    .kpi-sub {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.75rem;
        color: #6070A0;
        margin-top: 0.2rem;
    }

    /* Section header */
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: #00D4FF;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
    }

    .section-header-purple {
        font-family: 'Orbitron', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: #BF00FF;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(191, 0, 255, 0.3);
    }

    .section-header-green {
        font-family: 'Orbitron', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: #00FF88;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(0, 255, 136, 0.3);
    }

    .section-header-orange {
        font-family: 'Orbitron', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: #FF6B35;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(255, 107, 53, 0.3);
    }

    /* Body text */
    .body-text {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        color: #B0C0E0;
        line-height: 1.6;
    }

    /* Highlight box */
    .highlight-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(191, 0, 255, 0.08) 100%);
        border-left: 3px solid #00D4FF;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
    }

    /* Divider */
    .neon-divider {
        border: none;
        border-top: 1px solid rgba(0, 212, 255, 0.2);
        margin: 1rem 0;
    }

    /* Alert / warning */
    .warning-alert {
        background: rgba(255, 107, 53, 0.1);
        border: 1px solid rgba(255, 107, 53, 0.3);
        border-radius: 8px;
        padding: 0.75rem;
        color: #FF8855;
        font-family: 'Rajdhani', sans-serif;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0A0E27; }
    ::-webkit-scrollbar-thumb { background: #2A3A8A; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3A4AAA; }

    /* Streamlit overrides */
    .stMetric { background: transparent !important; }
    .stMetric label { color: #8090C0 !important; font-family: 'Rajdhani', sans-serif !important; }
    .stMetric [data-testid="stMetricValue"] { color: #00D4FF !important; font-family: 'Orbitron', monospace !important; }
    .stDataFrame { border: 1px solid rgba(0,212,255,0.1) !important; border-radius: 8px !important; }
    .stDataFrame [data-testid="stTable"] { font-family: 'Rajdhani', sans-serif !important; }
    div[data-testid="stExpander"] { border: 1px solid rgba(0,212,255,0.1) !important; border-radius: 8px !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(10, 14, 39, 0.6);
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 8px 8px 0 0;
        color: #8090C0;
        font-family: 'Rajdhani', sans-serif;
    }
    .stTabs [data-baseweb="tab"]:hover { background: rgba(0,212,255,0.05); color: #00D4FF; }
    .stTabs [aria-selected="true"] {
        background: rgba(0,212,255,0.1) !important;
        border-bottom: 2px solid #00D4FF;
        color: #00D4FF !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0A1A4A 0%, #1A2A6A 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: #00D4FF;
        border-radius: 8px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0A2A6A 0%, #1A3A8A 100%);
        border-color: rgba(0, 212, 255, 0.6);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
        color: #40E0FF;
    }

    /* Selectbox */
    .stSelectbox > div > div { background: rgba(10, 14, 39, 0.8) !important; border-color: rgba(0,212,255,0.2) !important; }

    /* Number input */
    .stNumberInput > div > div { background: rgba(10, 14, 39, 0.8) !important; border-color: rgba(0,212,255,0.2) !important; }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00D4FF, #BF00FF) !important;
    }

    /* Column headers in tables */
    thead th { background: rgba(0, 212, 255, 0.08) !important; color: #00D4FF !important; font-family: 'Rajdhani', sans-serif !important; }
    tbody tr:nth-child(even) { background: rgba(0, 212, 255, 0.03) !important; }
    </style>
    """, unsafe_allow_html=True)

def card(html_class, content):
    return f'<div class="{html_class}">{content}</div>'

def kpi_card(value, label, sub=None, color=NEON_BLUE):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ''
    return f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color}; text-shadow: 0 0 15px {color}66;">{value}</div>
        <div class="kpi-label">{label}</div>
        {sub_html}
    </div>
    """

def section_header(text, variant="blue"):
    cls = {
        "blue": "section-header",
        "purple": "section-header-purple",
        "green": "section-header-green",
        "orange": "section-header-orange",
    }.get(variant, "section-header")
    return f'<div class="{cls}">{text}</div>'

def glass_card(content):
    return f'<div class="glass-card">{content}</div>'

def highlight_box(text):
    return f'<div class="highlight-box"><span class="body-text">{text}</span></div>'

def render_html(html):
    st.markdown(html, unsafe_allow_html=True)

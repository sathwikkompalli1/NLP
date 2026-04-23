"""
IQAS Streamlit Application — Main Entrypoint
==============================================
Multi-page app with sidebar navigation, session state management, and CSS injection.
"""

import sys
from pathlib import Path

# ── Add project root to Python path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.ui.components import load_css, render_sidebar_brand
from utils.config import APP_TITLE, APP_ICON, APP_LAYOUT


# ── Page Configuration ──
st.set_page_config(
    page_title="IntelliRetrieve AI — Intelligent Document Retrieval & QA",
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Inject Custom CSS ──
load_css()

# ── Initialize Pipeline in Session State ──
if "pipeline" not in st.session_state:
    from core.pipeline import QAPipeline
    pipeline = QAPipeline()
    # Try to load existing index
    pipeline.load_index()
    st.session_state["pipeline"] = pipeline

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "query_log" not in st.session_state:
    st.session_state["query_log"] = []


# ── Sidebar ──
with st.sidebar:
    render_sidebar_brand()

    st.markdown("---")

    # Navigation
    st.markdown(
        '<p style="color: #9AA0A6; font-size: 0.8rem; text-transform: uppercase; '
        'letter-spacing: 0.1em; margin-bottom: 8px;">Navigation</p>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Go to:",
        options=["🧠 Workspace", "📊 Analytics", "🌐 Knowledge Graph"],
        index=0,
        label_visibility="collapsed",
        key="nav_radio",
    )


    # Footer — pinned to bottom of sidebar
    st.markdown(
        '<style>'
        '[data-testid="stSidebar"] > div:first-child {'
        '    display: flex; flex-direction: column; min-height: 100vh;'
        '}'
        '[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {'
        '    flex: 1; display: flex; flex-direction: column;'
        '}'
        '</style>'
        '<div style="margin-top: auto; text-align: center; color: #666; font-size: 0.75rem; '
        'padding: 16px 0; border-top: 1px solid rgba(255,255,255,0.08);">'
        '<p style="margin: 0;">IntelliRetrieve AI v1.0</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Page Routing ──
if page == "🧠 Workspace":
    from app.views.workspace import render_workspace_page
    render_workspace_page()

elif page == "📊 Analytics":
    from app.views.analytics import render_analytics_page
    render_analytics_page()

elif page == "🌐 Knowledge Graph":
    from app.views.knowledge import render_knowledge_page
    render_knowledge_page()

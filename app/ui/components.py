"""
IQAS UI Components
===================
Reusable Streamlit UI widgets for the IQAS application.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional


def load_css():
    """Inject custom CSS into the Streamlit app."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar_brand():
    """Render the sidebar branding area."""
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🧠 IntelliRetrieve AI</h2>
        <p>Intelligent Question Answering System</p>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(is_online: bool, label: str = ""):
    """Render a status badge (online/offline)."""
    status_class = "online" if is_online else "offline"
    dot = "🟢" if is_online else "🔴"
    text = label or ("System Ready" if is_online else "No Index Loaded")
    st.markdown(
        f'<div class="status-badge {status_class}">{dot} {text}</div>',
        unsafe_allow_html=True,
    )


def render_answer_card(answer_text: str):
    """Render a styled answer card."""
    st.markdown(f"""
    <div class="answer-card animate-fade-in">
        <div class="answer-text">{answer_text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_badge(confidence: float) -> str:
    """
    Render a confidence badge with color coding.

    Returns:
        HTML string for the confidence badge.
    """
    if confidence >= 0.8:
        css_class = "confidence-high"
        label = "High Confidence"
        icon = "✅"
    elif confidence >= 0.5:
        css_class = "confidence-med"
        label = "Medium Confidence"
        icon = "⚠️"
    else:
        css_class = "confidence-low"
        label = "Low Confidence"
        icon = "⚡"

    return (
        f'<div class="confidence-badge {css_class}">'
        f'{icon} {label} ({confidence:.1%})</div>'
    )


def render_source_chip(source: str, page: Optional[int] = None):
    """Render a source citation chip."""
    page_text = f" · Page {page}" if page else ""
    st.markdown(
        f'<div class="source-chip">📄 {source}{page_text}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(value: str, label: str):
    """Render a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_metrics_row(metrics: List[Dict[str, str]]):
    """
    Render a row of metric cards.

    Args:
        metrics: List of {'value': str, 'label': str} dicts.
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            st.markdown(
                render_metric_card(metric["value"], metric["label"]),
                unsafe_allow_html=True,
            )


def render_nlp_table(data: List[Dict[str, str]], headers: List[str]):
    """
    Render a styled NLP breakdown table.

    Args:
        data: List of row dicts.
        headers: Column header names.
    """
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = ""
    for row in data:
        cells = "".join(f"<td>{row.get(h, '')}</td>" for h in headers)
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <table class="nlp-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)


def render_section_header(title: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def render_chat_item(question: str, answer: str, confidence: float = 0.0):
    """Render a chat history item."""
    badge_html = render_confidence_badge(confidence) if confidence > 0 else ""
    st.markdown(f"""
    <div class="chat-item animate-fade-in">
        <div class="chat-question">❓ {question}</div>
        <div class="chat-answer">{answer}</div>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)

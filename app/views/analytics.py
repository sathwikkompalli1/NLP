"""
IQAS Analytics Page
====================
Query analytics, question type distribution, entity frequency, and export.
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui.components import (
    render_section_header,
    render_metrics_row,
)


def render_analytics_page():
    """Render the query analytics dashboard page."""

    st.markdown("# 📊 Analytics Dashboard")
    st.markdown(
        '<p style="color: #9AA0A6; font-size: 1.05rem;">'
        "Insights from your workspace sessions — question patterns, confidence trends, and more.</p>",
        unsafe_allow_html=True,
    )

    query_log = st.session_state.get("query_log", [])

    if not query_log:
        st.info(
            "📭 No queries yet. Go to the **💬 Q&A** page to ask questions, "
            "then come back here to see analytics."
        )
        return

    # ── Summary Metrics ──
    total_queries = len(query_log)
    avg_confidence = sum(q["confidence"] for q in query_log) / total_queries
    unique_types = len(set(q["question_type"] for q in query_log))
    all_entities = [e for q in query_log for e in q.get("entities", [])]
    unique_entities = len(set(all_entities))

    render_metrics_row([
        {"value": str(total_queries), "label": "Total Queries"},
        {"value": f"{avg_confidence:.1%}", "label": "Avg Confidence"},
        {"value": str(unique_types), "label": "Question Types"},
        {"value": str(unique_entities), "label": "Unique Entities"},
    ])

    st.markdown("---")

    # ── Question Type Distribution ──
    render_section_header("📊 Question Type Distribution")

    try:
        import plotly.express as px

        type_counts = {}
        for q in query_log:
            qt = q["question_type"]
            type_counts[qt] = type_counts.get(qt, 0) + 1

        type_df = pd.DataFrame([
            {"Question Type": k, "Count": v}
            for k, v in type_counts.items()
        ])

        fig = px.pie(
            type_df,
            values="Count",
            names="Question Type",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.35,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EAED",
            legend=dict(font=dict(color="#E8EAED")),
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback if plotly not available
        st.bar_chart(
            pd.DataFrame(type_counts.items(), columns=["Type", "Count"]).set_index("Type")
        )

    # ── Queries per Session (Timeline) ──
    render_section_header("📈 Confidence Timeline")

    conf_data = pd.DataFrame([
        {"Query #": i + 1, "Confidence": q["confidence"]}
        for i, q in enumerate(query_log)
    ])

    try:
        import plotly.express as px

        fig = px.line(
            conf_data,
            x="Query #",
            y="Confidence",
            markers=True,
            color_discrete_sequence=["#00B4D8"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EAED",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 1]),
            margin=dict(t=30, b=30),
        )
        fig.add_hline(y=0.8, line_dash="dot", line_color="#2ECC71", annotation_text="High Threshold")
        fig.add_hline(y=0.5, line_dash="dot", line_color="#F39C12", annotation_text="Med Threshold")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(conf_data.set_index("Query #"))

    # ── Document Coverage ──
    render_section_header("📚 Document Coverage")

    source_counts = {}
    for q in query_log:
        src = q.get("source", "Unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    if source_counts:
        source_df = pd.DataFrame([
            {"Document": k, "Times Retrieved": v}
            for k, v in source_counts.items()
        ])

        try:
            import plotly.express as px

            fig = px.bar(
                source_df,
                x="Document",
                y="Times Retrieved",
                color="Times Retrieved",
                color_continuous_scale=["#1E3A5F", "#00B4D8"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EAED",
                margin=dict(t=30, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(source_df.set_index("Document"))

    # ── Query Log Table ──
    st.markdown("---")
    render_section_header("📋 Full Query Log")

    log_df = pd.DataFrame([
        {
            "Time": q["timestamp"],
            "Question": q["question"][:60] + ("..." if len(q["question"]) > 60 else ""),
            "Type": q["question_type"],
            "Confidence": f"{q['confidence']:.1%}",
            "Source": q.get("source", "N/A"),
        }
        for q in query_log
    ])

    st.dataframe(log_df, hide_index=True)

    # ── Export ──
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        csv = log_df.to_csv(index=False)
        st.download_button(
            "📥 Export as CSV",
            data=csv,
            file_name="iqas_analytics.csv",
            mime="text/csv",
        )

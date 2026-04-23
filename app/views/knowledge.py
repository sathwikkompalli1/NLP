"""
IQAS Knowledge Graph Explorer Page
=====================================
Interactive knowledge graph visualization with entity-relationship extraction,
triple tables, co-occurrence heatmaps, and entity filtering.
"""

import streamlit as st
from pathlib import Path
import sys
import math

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui.components import render_section_header, render_metrics_row, render_nlp_table


# ── Sample texts for demo ──
SAMPLE_TEXTS = {
    "🏛️ History Passage": (
        "Albert Einstein developed the theory of relativity at the University of Zurich. "
        "Einstein worked with Niels Bohr on quantum mechanics in Copenhagen. "
        "The Nobel Prize was awarded to Einstein in 1921 for the photoelectric effect. "
        "Marie Curie also received the Nobel Prize for her research in radioactivity. "
        "Curie conducted her groundbreaking experiments at the University of Paris. "
        "Both Einstein and Curie transformed modern physics in the twentieth century."
    ),
    "🏢 Business News": (
        "Apple CEO Tim Cook announced the new iPhone at the Steve Jobs Theater in Cupertino. "
        "Microsoft CEO Satya Nadella responded with the launch of Surface Pro in Seattle. "
        "Google parent company Alphabet reported strong revenue growth in Mountain View. "
        "Amazon expanded its AWS cloud services across Europe and Asia. "
        "Tim Cook praised Apple's partnership with TSMC for chip manufacturing in Taiwan. "
        "Nadella highlighted Microsoft's collaboration with OpenAI for artificial intelligence."
    ),
    "🔬 NLP Research": (
        "Google introduced the Transformer architecture in the Attention Is All You Need paper. "
        "BERT was developed by Jacob Devlin at Google Research. "
        "OpenAI released GPT models that use transformer-based language modeling. "
        "Stanford University created the GloVe word embedding algorithm. "
        "Tomas Mikolov invented Word2Vec at Google. "
        "Yann LeCun at Facebook AI Research advanced convolutional neural networks."
    ),
}


def render_knowledge_page():
    """Render the Knowledge Graph Explorer page."""

    # ── Page Header ──
    st.markdown(
        '<div class="feature-hero">'
        '<div class="feature-hero-icon">🌐</div>'
        '<h1 class="feature-hero-title">Knowledge Graph Explorer</h1>'
        '<p class="feature-hero-subtitle">'
        'Discover entities, relationships, and concept networks — '
        'extracted automatically from your documents.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # ── Text Input ──
    st.markdown("---")
    render_section_header("📝 Input Text")

    # Build source options from uploaded documents
    source_options = list(SAMPLE_TEXTS.keys())
    uploaded_docs = {}

    pipeline = st.session_state.get("pipeline")
    if pipeline and pipeline._is_indexed:
        chunks = pipeline._chunks_data
        if chunks:
            # Group chunks by source document
            for chunk in chunks:
                src = chunk.get("source", "Unknown")
                if src not in uploaded_docs:
                    uploaded_docs[src] = []
                uploaded_docs[src].append(chunk.get("text", ""))

            for doc_name in uploaded_docs:
                source_options.insert(0, f"📚 {doc_name}")

    col1, col2 = st.columns([1, 2])
    with col1:
        sample_choice = st.selectbox(
            "Select text source:",
            source_options,
            key="kg_sample",
        )

    if sample_choice.startswith("📚 "):
        # Pull text from uploaded document
        doc_name = sample_choice.replace("📚 ", "")
        doc_chunks = uploaded_docs.get(doc_name, [])
        input_text = " ".join(doc_chunks)

        st.markdown(
            f'<div style="background: rgba(0,180,216,0.08); border: 1px solid rgba(0,180,216,0.2); '
            f'border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">'
            f'<span style="color: #00B4D8; font-weight: 600;">📚 Using uploaded document:</span> '
            f'<span style="color: #E8EAED;">{doc_name}</span> '
            f'<span style="color: #9AA0A6;">({len(doc_chunks)} chunks, {len(input_text):,} chars)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        input_text = SAMPLE_TEXTS[sample_choice]

    # ── Build Graph Button ──
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        build_btn = st.button(
            "🌐 Build Graph",
            key="build_kg",
            use_container_width=True,
        )

    if not build_btn or not input_text or not input_text.strip():
        if build_btn:
            st.warning("Please enter some text to build the knowledge graph.")
        return

    # ── Build Graph ──
    with st.spinner("🧠 Extracting entities and relationships..."):
        try:
            from nlp.knowledge_graph import KnowledgeGraphBuilder
            builder = KnowledgeGraphBuilder()
            graph = builder.build_graph(input_text.strip())
        except Exception as e:
            st.error(f"Graph building failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # ── Summary Metrics ──
    st.markdown("---")
    render_section_header("📊 Graph Overview")

    render_metrics_row([
        {"value": str(len(graph.nodes)), "label": "Nodes (Entities)"},
        {"value": str(len(graph.edges)), "label": "Edges (Relations)"},
        {"value": str(len(graph.triples)), "label": "Triples Extracted"},
        {"value": str(len(graph.entity_types)), "label": "Unique Entities"},
    ])

    if not graph.nodes:
        st.info("🔍 No entities or relationships found. Try text with named entities (people, places, organizations).")
        return

    # ── Interactive Network Graph ──
    st.markdown("---")
    render_section_header("🕸️ Interactive Knowledge Graph")
    st.markdown(
        '<p style="color: #9AA0A6; font-size: 0.9rem; margin-top: -8px;">'
        'Hover over nodes and edges to explore connections.</p>',
        unsafe_allow_html=True,
    )

    # Use all nodes and edges (no filtering)
    filtered_nodes = graph.nodes
    filtered_node_ids = set(n.id for n in filtered_nodes)
    filtered_edges = [e for e in graph.edges
                      if e.source in filtered_node_ids and e.target in filtered_node_ids]

    if filtered_nodes:
        _render_network_graph(filtered_nodes, filtered_edges)
    else:
        st.info("No nodes found in the graph.")

    # ── Entity Type Legend ──
    from nlp.knowledge_graph import ENTITY_TYPE_COLORS
    legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;">'
    for etype in sorted(set(n.entity_type for n in filtered_nodes)):
        color = ENTITY_TYPE_COLORS.get(etype, "#B2BEC3")
        legend_html += (
            f'<span style="background: {color}22; color: {color}; padding: 4px 12px; '
            f'border-radius: 20px; font-size: 0.8rem; font-weight: 600; '
            f'border: 1px solid {color}44;">● {etype}</span>'
        )
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

    # ── Extracted Triples Table ──
    st.markdown("---")
    render_section_header("🔗 Extracted Relationships (Triples)")
    st.markdown(
        '<p style="color: #9AA0A6; font-size: 0.9rem; margin-top: -8px;">'
        'Subject → Relation → Object triples extracted via dependency parsing.</p>',
        unsafe_allow_html=True,
    )

    if graph.triples:
        for triple in graph.triples:
            conf_color = "#2ECC71" if triple.confidence > 0.7 else (
                "#F39C12" if triple.confidence > 0.5 else "#E74C3C"
            )
            st.markdown(
                f'<div style="display: flex; align-items: center; gap: 8px; '
                f'padding: 12px 16px; background: #1A1D29; border-radius: 10px; '
                f'margin-bottom: 6px; border: 1px solid rgba(255,255,255,0.05);">'
                f'<span style="background: rgba(0,180,216,0.15); color: #00B4D8; '
                f'padding: 4px 12px; border-radius: 6px; font-weight: 600; '
                f'font-size: 0.85rem;">{triple.subject}</span>'
                f'<span style="color: #F39C12; font-size: 1.2rem;">→</span>'
                f'<span style="background: rgba(243,156,18,0.15); color: #F39C12; '
                f'padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; '
                f'font-style: italic;">{triple.relation}</span>'
                f'<span style="color: #F39C12; font-size: 1.2rem;">→</span>'
                f'<span style="background: rgba(46,204,113,0.15); color: #2ECC71; '
                f'padding: 4px 12px; border-radius: 6px; font-weight: 600; '
                f'font-size: 0.85rem;">{triple.object}</span>'
                f'<span style="margin-left: auto; color: {conf_color}; font-size: 0.75rem; '
                f'font-weight: 600;">{triple.confidence:.0%}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No triples could be extracted. Try text with clear subject-verb-object sentences.")

    # ── Entity Frequency Chart ──
    st.markdown("---")
    render_section_header("📊 Entity Frequency")

    if graph.entity_frequencies:
        try:
            import plotly.graph_objects as go

            sorted_entities = sorted(
                graph.entity_frequencies.items(),
                key=lambda x: x[1], reverse=True,
            )[:15]

            names = [e[0] for e in sorted_entities]
            counts = [e[1] for e in sorted_entities]
            colors = [
                ENTITY_TYPE_COLORS.get(graph.entity_types.get(n, "OTHER"), "#B2BEC3")
                for n in names
            ]

            fig = go.Figure(go.Bar(
                x=counts[::-1],
                y=names[::-1],
                orientation="h",
                marker=dict(
                    color=colors[::-1],
                    line=dict(width=0),
                    cornerradius=4,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Mentions: %{x}<br>"
                    "<extra></extra>"
                ),
            ))

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EAED",
                xaxis=dict(
                    title="Frequency",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(t=20, b=40, l=120, r=20),
                height=max(250, len(names) * 32),
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            import pandas as pd
            df = pd.DataFrame(
                list(graph.entity_frequencies.items()),
                columns=["Entity", "Count"],
            )
            st.bar_chart(df.set_index("Entity"))

    # ── Co-occurrence Matrix ──
    if graph.cooccurrence_matrix:
        st.markdown("---")
        render_section_header("🔥 Entity Co-occurrence Heatmap")
        st.markdown(
            '<p style="color: #9AA0A6; font-size: 0.9rem; margin-top: -8px;">'
            'Which entities appear together in the same sentences.</p>',
            unsafe_allow_html=True,
        )

        try:
            import plotly.graph_objects as go

            # Build matrix
            all_entities = sorted(set(
                list(graph.cooccurrence_matrix.keys()) +
                [e for d in graph.cooccurrence_matrix.values() for e in d.keys()]
            ))

            if len(all_entities) <= 20:
                matrix = []
                for e1 in all_entities:
                    row = []
                    for e2 in all_entities:
                        row.append(graph.cooccurrence_matrix.get(e1, {}).get(e2, 0))
                    matrix.append(row)

                fig = go.Figure(go.Heatmap(
                    z=matrix,
                    x=all_entities,
                    y=all_entities,
                    colorscale=[
                        [0, "#0F1117"],
                        [0.5, "#1E3A5F"],
                        [1, "#00B4D8"],
                    ],
                    hovertemplate=(
                        "<b>%{x}</b> ↔ <b>%{y}</b><br>"
                        "Co-occurrences: %{z}<extra></extra>"
                    ),
                ))

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E8EAED",
                    xaxis=dict(tickangle=45),
                    margin=dict(t=20, b=80, l=100, r=20),
                    height=max(300, len(all_entities) * 35),
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Too many entities for heatmap. Showing top co-occurring pairs.")
                pairs = []
                for e1, neighbors in graph.cooccurrence_matrix.items():
                    for e2, count in neighbors.items():
                        if e1 < e2:
                            pairs.append((e1, e2, count))
                pairs.sort(key=lambda x: x[2], reverse=True)

                import pandas as pd
                df = pd.DataFrame(pairs[:20], columns=["Entity A", "Entity B", "Co-occurrences"])
                st.dataframe(df, hide_index=True)

        except ImportError:
            st.info("Install plotly for heatmap visualization.")

    # ── Source Sentences ──
    if graph.triples:
        with st.expander("📖 Source Sentences", expanded=False):
            seen = set()
            for triple in graph.triples:
                if triple.sentence not in seen:
                    seen.add(triple.sentence)
                    st.markdown(
                        f'<div style="padding: 8px 12px; background: #1A1D29; '
                        f'border-radius: 8px; margin-bottom: 4px; color: #E8EAED; '
                        f'font-size: 0.9rem; line-height: 1.6;">{triple.sentence}</div>',
                        unsafe_allow_html=True,
                    )


def _render_network_graph(nodes, edges):
    """Render an interactive network graph using plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly required for network visualization.")
        return

    if not nodes:
        return

    # Assign positions using a simple force-directed-like layout
    n = len(nodes)
    node_positions = {}

    # Circular layout with some randomness based on frequency
    for i, node in enumerate(nodes):
        angle = (2 * math.pi * i) / n
        radius = 1.5 + (0.5 / max(node.frequency, 1))
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        node_positions[node.id] = (x, y)

    # ── Draw edges ──
    edge_traces = []
    edge_annotations = []

    for edge in edges:
        if edge.source not in node_positions or edge.target not in node_positions:
            continue

        x0, y0 = node_positions[edge.source]
        x1, y1 = node_positions[edge.target]

        opacity = 0.3 + edge.weight * 0.5

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(
                width=1 + edge.weight * 3,
                color=f"rgba(0, 180, 216, {opacity})",
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Edge label at midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if edge.relation != "co-occurs with":
            edge_annotations.append(dict(
                x=mx, y=my,
                text=edge.relation,
                showarrow=False,
                font=dict(size=9, color="rgba(243,156,18,0.7)"),
            ))

    # ── Draw nodes ──
    node_x = [node_positions[n.id][0] for n in nodes]
    node_y = [node_positions[n.id][1] for n in nodes]
    node_colors = [n.color for n in nodes]
    node_sizes = [max(20, min(50, 15 + n.frequency * 8)) for n in nodes]
    node_labels = [n.label for n in nodes]
    node_types = [n.entity_type for n in nodes]
    node_freqs = [n.frequency for n in nodes]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color="#1A1D29"),
            opacity=0.9,
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=11, color="#E8EAED"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Type: %{customdata[0]}<br>"
            "Mentions: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
        customdata=list(zip(node_types, node_freqs)),
        showlegend=False,
    )

    # ── Assemble figure ──
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            showline=False,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            showline=False,
        ),
        margin=dict(t=20, b=20, l=20, r=20),
        height=500,
        annotations=edge_annotations,
        font_color="#E8EAED",
        hoverlabel=dict(
            bgcolor="#1A1D29",
            bordercolor="#00B4D8",
            font_color="#E8EAED",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

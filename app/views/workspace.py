"""
IQAS Workspace Page
====================
Unified interface: question input with inline document upload,
followed by uploaded files preview and chunking configuration.
"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui.components import (
    render_section_header,
    render_metrics_row,
    render_nlp_table,
    render_answer_card,
    render_confidence_badge,
    render_source_chip,
    render_chat_item,
)
from utils.config import UPLOAD_DIR


def render_workspace_page():
    """Render the unified workspace page."""

    st.markdown("# 🧠 IntelliRetrieve AI Workspace")
    st.markdown(
        '<p style="color: #9AA0A6; font-size: 1.05rem;">'
        "Upload documents and ask questions — all in one place.</p>",
        unsafe_allow_html=True,
    )

    pipeline = st.session_state.get("pipeline")
    is_ready = pipeline.is_ready if pipeline else False

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: Unified Input — Upload + Question together
    # ═══════════════════════════════════════════════════════════════

    # File upload
    uploaded_files = st.file_uploader(
        "📎 Attach documents (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload documents to build a searchable knowledge base",
        key="doc_uploader",
        label_visibility="visible",
    )

    # Save uploaded files immediately so index can be built
    saved_paths = []
    if uploaded_files:
        for file in uploaded_files:
            save_path = UPLOAD_DIR / file.name
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(save_path)

    # Build Search Index button — only enabled when files are attached
    col_build1, col_build2, col_build3 = st.columns([1, 2, 1])
    with col_build2:
        build_clicked = st.button(
            "🚀 Build Search Index",
            key="build_index",
            use_container_width=True,
            disabled=not uploaded_files,
        )

    if build_clicked and saved_paths:
        _process_and_index(saved_paths, st.session_state.get("chunk_strategy", "sentence"))

    st.markdown("---")

    # Question input
    question = st.text_input(
        "Your Question:",
        placeholder="e.g., What is tokenization in NLP?" if is_ready else "Upload & index documents first, then ask questions...",
        key="question_input",
        disabled=not is_ready,
    )

    # Action buttons row
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        ask_button = st.button(
            "🔍 Ask",
            key="ask_button",
            use_container_width=True,
            disabled=not is_ready,
        )
    with col2:
        if st.button("🗑️ Clear", key="clear_history", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()
    with col3:
        if is_ready:
            stats = pipeline.get_stats()
            st.markdown(
                f'<div style="text-align: right; padding-top: 6px; color: #9AA0A6; font-size: 0.82rem;">'
                f'📚 {stats["num_documents"]} docs &nbsp;·&nbsp; '
                f'🧩 {stats["num_chunks"]} chunks &nbsp;·&nbsp; '
                f'🔍 {stats["index_size"]} vectors</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align: right; padding-top: 6px; color: #666; font-size: 0.82rem;">'
                '⚠️ No index built yet — upload documents and build index</div>',
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════════
    # Process question
    # ═══════════════════════════════════════════════════════════════
    if ask_button and question.strip() and is_ready:
        _handle_question(pipeline, question.strip())
    elif ask_button and not question.strip():
        st.warning("Please enter a question.")

    # ═══════════════════════════════════════════════════════════════
    # Display last answer details (persists across reruns)
    # ═══════════════════════════════════════════════════════════════
    last = st.session_state.get("last_answer")
    if last:
        st.markdown("---")
        render_section_header("📌 Answer")
        render_answer_card(last["answer"])

        # Confidence + Source + Time
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(
                render_confidence_badge(last["confidence"]),
                unsafe_allow_html=True,
            )
        with col2:
            render_source_chip(last["source"], last["page"])
        with col3:
            st.markdown(
                f'<div style="color: #9AA0A6; font-size: 0.85rem; padding-top: 6px;">'
                f'⏱️ {last["elapsed"]:.2f}s</div>',
                unsafe_allow_html=True,
            )

        # Supporting passage
        with st.expander("📖 Supporting Passage", expanded=False):
            st.markdown(
                f'<div style="background: #1A1D29; padding: 16px; border-radius: 8px; '
                f'line-height: 1.7; color: #E8EAED;">{last["supporting_passage"]}</div>',
                unsafe_allow_html=True,
            )

        # NLP Breakdown
        with st.expander("🧠 NLP Breakdown", expanded=False):
            st.markdown(f"**Question Type:** `{last['question_type']}`")

            try:
                from nlp.pos_tagger import POSTagger
                pos_tagger = POSTagger()

                tags = pos_tagger.get_detailed_tags(last["question"])
                pos_data = [
                    {"Word": word, "POS": pos, "Tag": tag}
                    for word, pos, tag in tags[:20]
                ]
                st.markdown("**POS Tags:**")
                render_nlp_table(pos_data, ["Word", "POS", "Tag"])

                if last.get("entities"):
                    st.markdown("**Entities Found:**")
                    for ent in last["entities"]:
                        st.markdown(f"- `{ent}`")

                keywords = pos_tagger.get_keywords(last["question"])
                if keywords:
                    st.markdown("**Keywords:** " + ", ".join(f"`{kw}`" for kw in keywords))

            except Exception as e:
                st.warning(f"NLP analysis error: {e}")

            if last.get("retrieval_scores"):
                st.markdown("**Top Passage Scores:**")
                score_data = [
                    {
                        "Chunk": s["chunk_id"][:12],
                        "Score": f"{s['score']:.4f}",
                        "Dense": f"{s['dense']:.4f}",
                        "BM25": f"{s['sparse']:.4f}",
                        "Rerank": f"{s['rerank']:.4f}",
                    }
                    for s in last["retrieval_scores"]
                ]
                render_nlp_table(score_data, ["Chunk", "Score", "Dense", "BM25", "Rerank"])

        # Entity highlighting
        with st.expander("🏷️ Entity Highlights", expanded=False):
            try:
                from nlp.ner import NERExtractor
                ner = NERExtractor()
                highlighted = ner.highlight_entities(last["answer"])
                st.markdown(highlighted, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Entity highlighting error: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Chat History
    # ═══════════════════════════════════════════════════════════════
    if st.session_state.get("chat_history"):
        st.markdown("---")
        render_section_header("💬 Chat History")
        history = list(reversed(st.session_state["chat_history"][-10:]))
        for item in history:
            render_chat_item(
                question=item["question"],
                answer=item["answer"],
                confidence=item.get("confidence", 0.0),
            )

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: Uploaded Files Preview
    # ═══════════════════════════════════════════════════════════════
    if saved_paths:
        st.markdown("---")
        render_section_header("📄 Uploaded Files")

        file_data = []
        for path in saved_paths:
            size_kb = path.stat().st_size / 1024
            file_data.append({
                "File": path.name,
                "Format": path.suffix.upper(),
                "Size": f"{size_kb:.1f} KB",
            })

        render_nlp_table(file_data, ["File", "Format", "Size"])

        # Text preview
        with st.expander("👁️ Preview Extracted Text", expanded=False):
            if pipeline:
                for path in saved_paths:
                    docs = pipeline.document_loader.load_any(path)
                    for doc in docs[:5]:
                        st.markdown(f"**{doc.filename}** (Page {doc.page_num})")
                        st.text(doc.text[:500] + ("..." if len(doc.text) > 500 else ""))
                        st.markdown("---")

        # NLP preview — aggregate text from ALL pages
        with st.expander("🧠 NLP Analysis Preview", expanded=False):
            if pipeline:
                for path in saved_paths[:1]:
                    docs = pipeline.document_loader.load_any(path)
                    if docs:
                        # Combine text from all pages (up to 5000 chars for preview)
                        full_text = " ".join(doc.text for doc in docs)
                        text = full_text[:5000]
                        try:
                            from nlp.tokenizer import NLPTokenizer
                            from nlp.ner import NERExtractor

                            tokenizer = NLPTokenizer()
                            ner = NERExtractor()

                            tokens = tokenizer.word_tokenize(text)
                            sentences = tokenizer.sent_tokenize(text)
                            entities = ner.extract(text)

                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                st.metric("Pages", len(docs))
                            with c2:
                                st.metric("Tokens", len(tokens))
                            with c3:
                                st.metric("Sentences", len(sentences))
                            with c4:
                                st.metric("Entities", len(entities))

                            if entities:
                                st.markdown("**Top Entities:**")
                                entity_data = [
                                    {"Entity": e.text, "Type": e.label}
                                    for e in entities[:10]
                                ]
                                render_nlp_table(entity_data, ["Entity", "Type"])
                        except Exception as e:
                            st.warning(f"NLP preview error: {e}")




# ──────────────────────── Helper Functions ────────────────────────


def _handle_question(pipeline, question: str):
    """Process a user question and save the answer to session state."""
    with st.spinner("🧠 Analyzing question and searching..."):
        start_time = time.time()
        answer = pipeline.ask(question)
        elapsed = time.time() - start_time

    # Save last answer to session state (rendered by main page)
    st.session_state["last_answer"] = {
        "question": question,
        "answer": answer.text,
        "confidence": answer.confidence,
        "source": answer.source,
        "page": answer.page,
        "question_type": answer.question_type,
        "supporting_passage": answer.supporting_passage,
        "entities": answer.entities,
        "retrieval_scores": answer.retrieval_scores,
        "elapsed": elapsed,
    }

    # Save to chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.session_state["chat_history"].append({
        "question": question,
        "answer": answer.text,
        "confidence": answer.confidence,
        "source": answer.source,
        "page": answer.page,
        "question_type": answer.question_type,
        "timestamp": time.strftime("%H:%M:%S"),
    })

    # Save to analytics
    if "query_log" not in st.session_state:
        st.session_state["query_log"] = []

    st.session_state["query_log"].append({
        "question": question,
        "question_type": answer.question_type,
        "confidence": answer.confidence,
        "source": answer.source,
        "entities": answer.entities,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def _process_and_index(paths, strategy):
    """Process documents and build the FAISS index."""
    pipeline = st.session_state.get("pipeline")
    if not pipeline:
        st.error("Pipeline not initialized. Please refresh the page.")
        return

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_callback(stage: str, progress: float):
        progress_bar.progress(progress)
        status_text.markdown(
            f'<div style="color: #00B4D8; font-weight: 600;">⏳ {stage}</div>',
            unsafe_allow_html=True,
        )

    try:
        stats = pipeline.ingest_documents(
            paths=[str(p) for p in paths],
            strategy=strategy,
            progress_callback=progress_callback,
        )

        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()

        st.markdown(
            f'<div style="background: rgba(46,204,113,0.1); border: 1px solid rgba(46,204,113,0.3); '
            f'border-radius: 10px; padding: 10px 16px; font-size: 0.88rem; color: #2ECC71;">'
            f'✅ Indexed <strong>{stats.num_documents}</strong> doc → '
            f'<strong>{stats.num_chunks}</strong> chunks · '
            f'{stats.total_tokens:,} tokens · '
            f'{stats.index_time_seconds}s'
            f'</div>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        progress_bar.progress(0.0)
        status_text.empty()
        st.error(f"❌ Indexing failed: {e}")
        import traceback
        st.code(traceback.format_exc())

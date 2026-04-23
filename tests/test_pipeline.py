"""
IQAS Tests — Pipeline
======================
End-to-end tests for document ingestion and question answering.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


class TestPipeline:
    """End-to-end pipeline tests."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        """Initialize pipeline with sample data."""
        from core.pipeline import QAPipeline

        self.pipeline = QAPipeline()
        self.sample_path = PROJECT_ROOT / "data" / "sample_docs" / "sample_nlp_notes.txt"

        if self.sample_path.exists():
            self.pipeline.ingest_documents(
                paths=[str(self.sample_path)],
                strategy="sentence",
            )

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "sample_nlp_notes.txt").exists(),
        reason="Sample data file not found",
    )
    def test_ingest_and_ask(self):
        """End-to-end test: ingest sample docs, then ask a question."""
        assert self.pipeline.is_ready

        answer = self.pipeline.ask("What is tokenization?")
        assert answer.text
        assert len(answer.text) > 10
        # Answer should mention tokenization-related content
        assert any(
            kw in answer.text.lower()
            for kw in ["token", "split", "text", "word", "break"]
        )

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "sample_nlp_notes.txt").exists(),
        reason="Sample data file not found",
    )
    def test_answer_has_source(self):
        """Verify source citation is present in answer."""
        answer = self.pipeline.ask("What is NLP?")
        assert answer.source
        assert answer.source != "N/A"
        # Source should reference the sample file
        assert "sample" in answer.source.lower() or answer.source.endswith(".txt")

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "sample_nlp_notes.txt").exists(),
        reason="Sample data file not found",
    )
    def test_confidence_range(self):
        """Verify confidence is in [0, 1] range."""
        answer = self.pipeline.ask("Who developed Word2Vec?")
        assert 0.0 <= answer.confidence <= 1.0

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "sample_nlp_notes.txt").exists(),
        reason="Sample data file not found",
    )
    def test_question_type_detection(self):
        """Verify question type is set correctly."""
        answer = self.pipeline.ask("Who developed Word2Vec?")
        assert answer.question_type == "WHO"

        answer = self.pipeline.ask("What is NER?")
        assert answer.question_type in ("DEFINE", "WHAT")

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "sample_nlp_notes.txt").exists(),
        reason="Sample data file not found",
    )
    def test_get_stats(self):
        """Verify stats are populated after ingestion."""
        stats = self.pipeline.get_stats()
        assert stats["is_indexed"] is True
        assert stats["num_chunks"] > 0
        assert stats["index_size"] > 0

    def test_ask_without_index(self):
        """Verify graceful handling when no index is loaded."""
        from core.pipeline import QAPipeline
        empty_pipeline = QAPipeline()
        answer = empty_pipeline.ask("Test question?")
        assert "no documents" in answer.text.lower() or "index" in answer.text.lower()

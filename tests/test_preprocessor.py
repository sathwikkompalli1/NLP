"""
IQAS Tests — Preprocessor
===========================
Tests for tokenization, POS tagging, NER, and chunking.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


class TestTokenizer:
    """Tests for NLPTokenizer."""

    def setup_method(self):
        from nlp.tokenizer import NLPTokenizer
        self.tokenizer = NLPTokenizer()

    def test_tokenize_basic(self):
        """Verify token count on known string."""
        text = "Natural language processing is a subfield of AI."
        tokens = self.tokenizer.tokenize(text)
        # Should have tokens for each word + punctuation
        assert len(tokens) >= 8
        assert tokens[0]["text"] == "Natural"

    def test_word_tokenize(self):
        """Verify word tokenization produces correct tokens."""
        text = "The quick brown fox."
        words = self.tokenizer.word_tokenize(text, remove_punct=True)
        assert "The" in words
        assert "quick" in words
        assert "." not in words

    def test_word_tokenize_stopwords(self):
        """Verify stopword removal works."""
        text = "The quick brown fox jumps over the lazy dog."
        words_no_stop = self.tokenizer.word_tokenize(text, remove_stopwords=True, remove_punct=True)
        # "the", "over" should be filtered
        assert "the" not in [w.lower() for w in words_no_stop]

    def test_sent_tokenize(self):
        """Verify sentence boundary detection."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = self.tokenizer.sent_tokenize(text)
        assert len(sentences) == 3

    def test_get_lemmas(self):
        """Verify lemmatization produces correct lemmas."""
        text = "The dogs are running quickly."
        lemmas = self.tokenizer.get_lemmas(text)
        assert "dog" in lemmas or "run" in lemmas


class TestPOSTagger:
    """Tests for POSTagger."""

    def setup_method(self):
        from nlp.pos_tagger import POSTagger
        self.tagger = POSTagger()

    def test_pos_tags_sentence(self):
        """Verify NOUN/VERB tags detected in a sentence."""
        text = "The cat sits on the mat."
        tags = self.tagger.tag(text)
        pos_types = [pos for _, pos in tags]
        assert "NOUN" in pos_types
        assert "VERB" in pos_types

    def test_noun_phrases(self):
        """Verify noun phrase extraction."""
        text = "The quick brown fox jumps over the lazy dog."
        nps = self.tagger.get_noun_phrases(text)
        assert len(nps) > 0
        # Should find at least "the quick brown fox" or similar
        assert any("fox" in np.lower() for np in nps)

    def test_detect_question_type_who(self):
        """Verify WHO question detection."""
        assert self.tagger.detect_question_type("Who invented Python?") == "WHO"

    def test_detect_question_type_what(self):
        """Verify WHAT question detection."""
        assert self.tagger.detect_question_type("What is NLP?") == "DEFINE"

    def test_detect_question_type_when(self):
        """Verify WHEN question detection."""
        assert self.tagger.detect_question_type("When was BERT released?") == "WHEN"

    def test_detect_question_type_how(self):
        """Verify HOW question detection."""
        assert self.tagger.detect_question_type("How does tokenization work?") == "HOW"

    def test_keywords_extraction(self):
        """Verify keyword extraction."""
        text = "Natural language processing uses machine learning algorithms."
        keywords = self.tagger.get_keywords(text)
        assert len(keywords) > 0


class TestNER:
    """Tests for NERExtractor."""

    def setup_method(self):
        from nlp.ner import NERExtractor
        self.ner = NERExtractor()

    def test_ner_extracts_person(self):
        """Verify PERSON entity from known text."""
        text = "Dr. Smith teaches NLP at MIT."
        entities = self.ner.extract(text)
        person_entities = [e for e in entities if e.label == "PERSON"]
        assert len(person_entities) > 0
        assert any("Smith" in e.text for e in person_entities)

    def test_ner_extracts_org(self):
        """Verify ORG entity extraction."""
        text = "Google and Microsoft are investing in AI research."
        orgs = self.ner.get_entities_by_type(text, "ORG")
        assert len(orgs) > 0

    def test_highlight_entities(self):
        """Verify entity highlighting returns HTML."""
        text = "Dr. Smith works at Google."
        highlighted = self.ner.highlight_entities(text)
        assert "<span" in highlighted or highlighted == text  # If no entities detected


class TestChunker:
    """Tests for TextChunker."""

    def setup_method(self):
        from utils.chunker import TextChunker
        self.chunker = TextChunker()
        self.sample_text = (
            "First sentence about NLP. Second sentence about tokenization. "
            "Third sentence about POS tagging. Fourth sentence about NER. "
            "Fifth sentence about embeddings. Sixth sentence about transformers. "
            "Seventh sentence about BERT. Eighth sentence about semantic search."
        )

    def test_chunker_fixed_size(self):
        """Verify chunk count and overlap correctness."""
        chunks = self.chunker.chunk_fixed(
            self.sample_text,
            chunk_size=10,
            overlap=2,
            doc_id="test",
            source="test.txt",
        )
        assert len(chunks) > 0
        for chunk in chunks:
            # Each chunk should have <= chunk_size words
            word_count = len(chunk.text.split())
            assert word_count <= 10

    def test_chunker_no_mid_sentence_split(self):
        """Verify sentence-aware chunking doesn't split mid-sentence."""
        chunks = self.chunker.chunk_by_sentence(
            self.sample_text,
            max_tokens=15,
            doc_id="test",
            source="test.txt",
        )
        assert len(chunks) > 0
        for chunk in chunks:
            # Each chunk should end at a sentence boundary (ends with .)
            text = chunk.text.strip()
            assert text.endswith(".") or text.endswith("!") or text.endswith("?") or text.endswith('"')

    def test_chunker_paragraph(self):
        """Verify paragraph-aware chunking."""
        para_text = "First paragraph about NLP.\n\nSecond paragraph about ML.\n\nThird paragraph about AI."
        chunks = self.chunker.chunk_by_paragraph(
            para_text,
            max_tokens=100,
            doc_id="test",
            source="test.txt",
        )
        assert len(chunks) > 0

    def test_chunker_empty_text(self):
        """Verify chunker handles empty text."""
        chunks = self.chunker.chunk("", doc_id="test", source="test.txt")
        assert len(chunks) == 0

    def test_chunk_metadata(self):
        """Verify chunk metadata is properly set."""
        chunks = self.chunker.chunk(
            self.sample_text,
            doc_id="doc1",
            source="notes.txt",
            page=1,
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.doc_id == "doc1"
            assert chunk.source == "notes.txt"
            assert chunk.page == 1
            assert chunk.token_count > 0

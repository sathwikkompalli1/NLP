"""
IQAS Smart Text Chunking
=========================
Three configurable strategies for splitting documents into retrievable chunks.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import spacy

from utils.config import CHUNK_SIZE, CHUNK_OVERLAP, SENTENCE_CHUNK_MAX_TOKENS, PARAGRAPH_CHUNK_MAX_TOKENS
from utils.logger import get_logger

log = get_logger("chunker")

# ──────────────────────────── Data Models ────────────────────────────


@dataclass
class Chunk:
    """A document chunk with full provenance metadata."""
    chunk_id: str
    text: str
    doc_id: str
    source: str
    page: Optional[int] = None
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())[:12]
        self.token_count = len(self.text.split())


# ──────────────────────────── Chunker ────────────────────────────


class TextChunker:
    """
    Smart text chunker with three strategies.

    Strategies:
        - fixed: Fixed token count with overlap
        - sentence: Sentence-aware grouping (no mid-sentence splits)
        - paragraph: Paragraph-aware splitting with size limits
    """

    def __init__(self):
        """Initialize with a shared spaCy model for sentence detection."""
        try:
            self._nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except OSError:
            log.warning("spaCy model not found — using blank English model for sentencizer")
            self._nlp = spacy.blank("en")
            self._nlp.add_pipe("sentencizer")
        # Increase max length for large documents
        self._nlp.max_length = 2_000_000

    def chunk(
        self,
        text: str,
        doc_id: str = "",
        source: str = "",
        page: Optional[int] = None,
        strategy: str = "sentence",
    ) -> List[Chunk]:
        """
        Chunk text using the specified strategy.

        Args:
            text: Full text to chunk.
            doc_id: Document identifier.
            source: Source filename.
            page: Page number (if applicable).
            strategy: One of 'fixed', 'sentence', 'paragraph'.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        strategy = strategy.lower()
        if strategy == "fixed":
            return self.chunk_fixed(text, doc_id=doc_id, source=source, page=page)
        elif strategy == "sentence":
            return self.chunk_by_sentence(text, doc_id=doc_id, source=source, page=page)
        elif strategy == "paragraph":
            return self.chunk_by_paragraph(text, doc_id=doc_id, source=source, page=page)
        else:
            log.warning(f"Unknown strategy '{strategy}', falling back to 'sentence'")
            return self.chunk_by_sentence(text, doc_id=doc_id, source=source, page=page)

    def chunk_fixed(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        doc_id: str = "",
        source: str = "",
        page: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Fixed-size chunking with token overlap.

        Splits text into chunks of `chunk_size` tokens with `overlap` token overlap.
        """
        words = text.split()
        if not words:
            return []

        chunks: List[Chunk] = []
        step = max(1, chunk_size - overlap)
        idx = 0

        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Calculate character offsets
            start_char = len(" ".join(words[:start])) + (1 if start > 0 else 0)
            end_char = start_char + len(chunk_text)

            chunks.append(Chunk(
                chunk_id=f"{doc_id}_c{idx}",
                text=chunk_text,
                doc_id=doc_id,
                source=source,
                page=page,
                start_char=start_char,
                end_char=end_char,
            ))
            idx += 1

            if end >= len(words):
                break

        log.debug(f"Fixed chunking: {len(chunks)} chunks from {len(words)} words")
        return chunks

    def chunk_by_sentence(
        self,
        text: str,
        max_tokens: int = SENTENCE_CHUNK_MAX_TOKENS,
        doc_id: str = "",
        source: str = "",
        page: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Sentence-aware chunking — groups sentences until token limit, never splits mid-sentence.
        """
        doc = self._nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_token_count = 0
        idx = 0

        for sent in sentences:
            sent_tokens = len(sent.split())

            # If a single sentence exceeds max_tokens, add it as its own chunk
            if sent_tokens > max_tokens:
                # Flush current buffer first
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    start_char = text.find(current_sentences[0])
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_c{idx}",
                        text=chunk_text,
                        doc_id=doc_id,
                        source=source,
                        page=page,
                        start_char=max(0, start_char),
                        end_char=max(0, start_char) + len(chunk_text),
                    ))
                    idx += 1
                    current_sentences = []
                    current_token_count = 0

                # Add the long sentence as its own chunk
                start_char = text.find(sent)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_c{idx}",
                    text=sent,
                    doc_id=doc_id,
                    source=source,
                    page=page,
                    start_char=max(0, start_char),
                    end_char=max(0, start_char) + len(sent),
                ))
                idx += 1
                continue

            # If adding this sentence would exceed the limit, flush
            if current_token_count + sent_tokens > max_tokens and current_sentences:
                chunk_text = " ".join(current_sentences)
                start_char = text.find(current_sentences[0])
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_c{idx}",
                    text=chunk_text,
                    doc_id=doc_id,
                    source=source,
                    page=page,
                    start_char=max(0, start_char),
                    end_char=max(0, start_char) + len(chunk_text),
                ))
                idx += 1
                current_sentences = []
                current_token_count = 0

            current_sentences.append(sent)
            current_token_count += sent_tokens

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start_char = text.find(current_sentences[0])
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_c{idx}",
                text=chunk_text,
                doc_id=doc_id,
                source=source,
                page=page,
                start_char=max(0, start_char),
                end_char=max(0, start_char) + len(chunk_text),
            ))

        log.debug(f"Sentence chunking: {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks

    def chunk_by_paragraph(
        self,
        text: str,
        max_tokens: int = PARAGRAPH_CHUNK_MAX_TOKENS,
        doc_id: str = "",
        source: str = "",
        page: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Paragraph-aware chunking — splits on double newlines, then size-limits.
        """
        # Split on double newlines (paragraph boundaries)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return []

        chunks: List[Chunk] = []
        current_paragraphs: List[str] = []
        current_token_count = 0
        idx = 0

        for para in paragraphs:
            para_tokens = len(para.split())

            # If a single paragraph exceeds max_tokens, chunk it by sentence
            if para_tokens > max_tokens:
                # Flush current buffer
                if current_paragraphs:
                    chunk_text = "\n\n".join(current_paragraphs)
                    start_char = text.find(current_paragraphs[0])
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_c{idx}",
                        text=chunk_text,
                        doc_id=doc_id,
                        source=source,
                        page=page,
                        start_char=max(0, start_char),
                        end_char=max(0, start_char) + len(chunk_text),
                    ))
                    idx += 1
                    current_paragraphs = []
                    current_token_count = 0

                # Sub-chunk the large paragraph by sentence
                sub_chunks = self.chunk_by_sentence(
                    para, max_tokens=max_tokens, doc_id=doc_id, source=source, page=page
                )
                for sc in sub_chunks:
                    sc.chunk_id = f"{doc_id}_c{idx}"
                    chunks.append(sc)
                    idx += 1
                continue

            if current_token_count + para_tokens > max_tokens and current_paragraphs:
                chunk_text = "\n\n".join(current_paragraphs)
                start_char = text.find(current_paragraphs[0])
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_c{idx}",
                    text=chunk_text,
                    doc_id=doc_id,
                    source=source,
                    page=page,
                    start_char=max(0, start_char),
                    end_char=max(0, start_char) + len(chunk_text),
                ))
                idx += 1
                current_paragraphs = []
                current_token_count = 0

            current_paragraphs.append(para)
            current_token_count += para_tokens

        # Flush remaining
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            start_char = text.find(current_paragraphs[0])
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_c{idx}",
                text=chunk_text,
                doc_id=doc_id,
                source=source,
                page=page,
                start_char=max(0, start_char),
                end_char=max(0, start_char) + len(chunk_text),
            ))

        log.debug(f"Paragraph chunking: {len(chunks)} chunks from {len(paragraphs)} paragraphs")
        return chunks

"""
IQAS POS Tagger
================
POS tagging, noun phrase extraction, keyword extraction, and question analysis.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

from nlp.tokenizer import NLPTokenizer
from utils.logger import get_logger

log = get_logger("pos_tagger")


class POSTagger:
    """
    POS tagging and question analysis using spaCy.

    Provides:
        - POS tag assignment
        - Noun phrase (NP chunk) extraction
        - Keyword extraction (NOUN + PROPN + key VERBs)
        - Question type detection (WHO/WHAT/WHEN/WHERE/WHY/HOW/DEFINE)
        - Question focus extraction
    """

    def __init__(self):
        """Initialize with shared spaCy tokenizer."""
        self._tokenizer = NLPTokenizer()
        self.nlp = self._tokenizer.nlp

    def tag(self, text: str) -> List[Tuple[str, str]]:
        """
        Assign POS tags to each token.

        Args:
            text: Input text.

        Returns:
            List of (word, POS_tag) tuples using Universal POS tags.
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc if not token.is_space]

    def get_detailed_tags(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Get detailed POS tags (word, universal POS, fine-grained tag).

        Returns:
            List of (word, pos, tag) triples.
        """
        doc = self.nlp(text)
        return [
            (token.text, token.pos_, token.tag_)
            for token in doc
            if not token.is_space
        ]

    def get_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases (NP chunks) from text.

        Args:
            text: Input text.

        Returns:
            List of noun phrase strings.
        """
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def get_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords — NOUN, PROPN, and key VERB lemmas.

        Args:
            text: Input text.
            top_n: Maximum number of keywords to return.

        Returns:
            List of keyword strings (deduplicated, ranked by frequency).
        """
        doc = self.nlp(text)
        keyword_pos = {"NOUN", "PROPN", "VERB"}

        # Collect lemmatized keywords
        keywords: dict[str, int] = {}
        for token in doc:
            if token.pos_ in keyword_pos and not token.is_stop and not token.is_punct:
                lemma = token.lemma_.lower()
                if len(lemma) > 1:  # Skip single-char tokens
                    keywords[lemma] = keywords.get(lemma, 0) + 1

        # Sort by frequency, return top_n
        sorted_kw = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_kw[:top_n]]

    def detect_question_type(self, question: str) -> str:
        """
        Detect the type of a question.

        Categories:
            WHO   → Person/organization questions
            WHAT  → Definition/fact questions
            WHEN  → Temporal questions
            WHERE → Location questions
            WHY   → Causal/reason questions
            HOW   → Process/method questions
            DEFINE → Definition requests
            OTHER → Unclassified

        Args:
            question: Question string.

        Returns:
            Question type string.
        """
        q_lower = question.lower().strip()

        # Check for explicit definition requests
        define_patterns = [
            "define ", "what is the definition of", "what does",
            "explain the term", "explain the concept", "what is meant by",
            "describe ", "what are ", "what is ",
        ]
        for pattern in define_patterns:
            if q_lower.startswith(pattern):
                return "DEFINE"

        # Check for WH-word at the start
        wh_map = {
            "who": "WHO",
            "whom": "WHO",
            "whose": "WHO",
            "what": "WHAT",
            "which": "WHAT",
            "when": "WHEN",
            "where": "WHERE",
            "why": "WHY",
            "how": "HOW",
        }

        first_word = q_lower.split()[0] if q_lower.split() else ""
        if first_word in wh_map:
            return wh_map[first_word]

        # Check for WH-word anywhere (for inverted questions)
        for wh_word, q_type in wh_map.items():
            if wh_word in q_lower.split():
                return q_type

        # Check for yes/no patterns
        yn_starters = ["is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "should", "has", "have"]
        if first_word in yn_starters:
            return "WHAT"

        return "OTHER"

    def extract_question_focus(self, question: str) -> List[str]:
        """
        Extract the key focus terms from a question.

        Combines noun phrases and NOUN/PROPN tokens to identify what the question is about.

        Args:
            question: Question string.

        Returns:
            List of focus term strings (deduplicated).
        """
        doc = self.nlp(question)

        focus_terms = []

        # Add noun phrases
        for chunk in doc.noun_chunks:
            # Skip chunks that are just WH-words
            if chunk.root.pos_ != "PRON" and not chunk.root.is_stop:
                focus_terms.append(chunk.text.lower())

        # Add standalone NOUN/PROPN not in chunks
        chunk_tokens = set()
        for chunk in doc.noun_chunks:
            for token in chunk:
                chunk_tokens.add(token.i)

        for token in doc:
            if token.i not in chunk_tokens and token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                focus_terms.append(token.lemma_.lower())

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for term in focus_terms:
            if term not in seen:
                seen.add(term)
                deduped.append(term)

        return deduped

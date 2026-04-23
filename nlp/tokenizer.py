"""
IQAS Tokenizer
===============
spaCy-based word and sentence tokenization with lemmatization.
"""

from __future__ import annotations

from typing import List, Optional

import spacy
from spacy.tokens import Token

from utils.logger import get_logger
from utils.config import SPACY_MODEL

log = get_logger("tokenizer")


class NLPTokenizer:
    """
    Custom spaCy tokenizer with word/sentence tokenization and lemmatization.

    The spaCy model is loaded once and shared across calls for efficiency.
    """

    _shared_nlp: Optional[spacy.language.Language] = None

    def __init__(self, model_name: str = SPACY_MODEL):
        """
        Initialize the tokenizer.

        Args:
            model_name: spaCy model to load (default: en_core_web_sm).
        """
        if NLPTokenizer._shared_nlp is None:
            try:
                NLPTokenizer._shared_nlp = spacy.load(model_name)
                log.info(f"Loaded spaCy model: {model_name}")
            except OSError:
                log.warning(f"Model '{model_name}' not found — using blank English model")
                NLPTokenizer._shared_nlp = spacy.blank("en")
                NLPTokenizer._shared_nlp.add_pipe("sentencizer")
            NLPTokenizer._shared_nlp.max_length = 2_000_000
        self.nlp = NLPTokenizer._shared_nlp

    def tokenize(self, text: str) -> List[dict]:
        """
        Full tokenization with all spaCy token attributes.

        Args:
            text: Input text.

        Returns:
            List of token dicts with text, lemma, pos, tag, dep, is_stop, is_punct.
        """
        doc = self.nlp(text)
        return [
            {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "is_stop": token.is_stop,
                "is_punct": token.is_punct,
                "is_space": token.is_space,
            }
            for token in doc
        ]

    def sent_tokenize(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy's sentence boundary detection.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def word_tokenize(self, text: str, remove_stopwords: bool = False, remove_punct: bool = True) -> List[str]:
        """
        Extract word tokens from text.

        Args:
            text: Input text.
            remove_stopwords: If True, filter out stop words.
            remove_punct: If True, filter out punctuation.

        Returns:
            List of word strings.
        """
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if token.is_space:
                continue
            if remove_punct and token.is_punct:
                continue
            if remove_stopwords and token.is_stop:
                continue
            tokens.append(token.text)
        return tokens

    def get_lemmas(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Extract lemmatized forms of words.

        Args:
            text: Input text.
            remove_stopwords: If True, filter out stop words.

        Returns:
            List of lemma strings.
        """
        doc = self.nlp(text)
        lemmas = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if remove_stopwords and token.is_stop:
                continue
            lemmas.append(token.lemma_.lower())
        return lemmas

    def get_token_count(self, text: str) -> int:
        """Return the number of non-whitespace tokens in text."""
        doc = self.nlp(text)
        return sum(1 for token in doc if not token.is_space)

    def get_sentence_count(self, text: str) -> int:
        """Return the number of sentences in text."""
        return len(self.sent_tokenize(text))

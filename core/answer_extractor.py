"""
IQAS Answer Extractor
======================
Extract precise answers from retrieved passages with confidence scoring and source citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.retriever import RetrievedChunk
from nlp.embedder import TextEmbedder
from nlp.ner import NERExtractor
from nlp.pos_tagger import POSTagger
from nlp.similarity import batch_cosine_similarity
from nlp.tokenizer import NLPTokenizer
from utils.logger import get_logger

log = get_logger("answer_extractor")


# ──────────────────────────── Data Model ────────────────────────────


@dataclass
class Answer:
    """A structured answer with provenance and confidence."""
    text: str
    source: str
    page: Optional[int]
    confidence: float
    supporting_passage: str
    entities: List[str] = field(default_factory=list)
    question_type: str = ""
    retrieval_scores: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "confidence": round(self.confidence, 4),
            "supporting_passage": self.supporting_passage,
            "entities": self.entities,
            "question_type": self.question_type,
        }


# ──────────────────────────── Answer Extractor ────────────────────────────


class AnswerExtractor:
    """
    Extract answers from retrieved passages.

    Process:
        1. Select best passage from top-K retrieved chunks
        2. Find most relevant sentences within the passage
        3. Extract answer span based on question type
        4. Compute confidence score
        5. Add source citation and entities
    """

    def __init__(self, embedder: TextEmbedder):
        """
        Initialize answer extractor.

        Args:
            embedder: Text embedder for sentence-level similarity.
        """
        self.embedder = embedder
        self._tokenizer = NLPTokenizer()
        self._ner = NERExtractor()
        self._pos_tagger = POSTagger()

    def extract_answer(
        self,
        question: str,
        passages: List[RetrievedChunk],
    ) -> Answer:
        """
        Extract the best answer from retrieved passages.

        Args:
            question: User question.
            passages: List of retrieved passage chunks (ranked by relevance).

        Returns:
            Answer object with text, source, confidence, and entities.
        """
        if not passages:
            return Answer(
                text="I couldn't find relevant information to answer this question.",
                source="N/A",
                page=None,
                confidence=0.0,
                supporting_passage="",
                question_type=self._pos_tagger.detect_question_type(question),
            )

        # Detect question type
        q_type = self._pos_tagger.detect_question_type(question)
        log.info(f"Question type: {q_type}")

        # Select best passage (first passage is highest ranked after re-ranking)
        best_passage = passages[0]

        # Find best sentences within the passage
        best_sentences = self.find_best_sentences(question, best_passage.text, n=3)

        # Construct answer based on question type
        if q_type in ("DEFINE", "WHAT"):
            # For definitional questions, include more context
            answer_text = " ".join(best_sentences[:3])
        elif q_type in ("WHO", "WHEN", "WHERE"):
            # For factoid questions, be more concise
            answer_text = " ".join(best_sentences[:2])
        else:
            answer_text = " ".join(best_sentences[:2])

        # Extract entities from answer
        entities = self._ner.get_answer_entities(answer_text, q_type)

        # Compute confidence
        confidence = self.compute_confidence(
            retrieval_score=best_passage.score,
            rerank_score=best_passage.rerank_score,
        )

        # Build retrieval scores for analytics
        retrieval_scores = [
            {
                "chunk_id": p.chunk_id,
                "score": round(p.score, 4),
                "dense": round(p.dense_score, 4),
                "sparse": round(p.sparse_score, 4),
                "rerank": round(p.rerank_score, 4),
            }
            for p in passages
        ]

        # Extract source filename only
        source_name = best_passage.source
        if "/" in source_name or "\\" in source_name:
            source_name = source_name.replace("\\", "/").split("/")[-1]

        return Answer(
            text=answer_text,
            source=source_name,
            page=best_passage.page,
            confidence=confidence,
            supporting_passage=best_passage.text,
            entities=entities,
            question_type=q_type,
            retrieval_scores=retrieval_scores,
        )

    def find_best_sentences(
        self,
        question: str,
        passage_text: str,
        n: int = 2,
    ) -> List[str]:
        """
        Find the most relevant sentences in a passage for a question.

        Uses sentence-level cosine similarity between the question
        and each sentence in the passage.

        Args:
            question: User question.
            passage_text: Full passage text.
            n: Number of best sentences to return.

        Returns:
            List of best sentence strings (in original order).
        """
        sentences = self._tokenizer.sent_tokenize(passage_text)

        if not sentences:
            return [passage_text[:500]]

        if len(sentences) <= n:
            return sentences

        # Compute sentence-level similarity
        question_vec = self.embedder.embed(question)
        sentence_vecs = self.embedder.embed_batch(sentences)

        similarities = batch_cosine_similarity(question_vec, sentence_vecs)

        # Get top-N indices sorted by similarity
        top_n = min(n, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_n]

        # Return in original order (preserve reading flow)
        top_indices_sorted = sorted(top_indices)
        return [sentences[i] for i in top_indices_sorted]

    def compute_confidence(
        self,
        retrieval_score: float,
        rerank_score: float,
    ) -> float:
        """
        Compute a confidence score from retrieval and re-ranking scores.

        Weighted average: 30% retrieval + 70% re-ranker (if available).
        Result is clamped to [0, 1].

        Args:
            retrieval_score: Score from hybrid retrieval (RRF).
            rerank_score: Score from cross-encoder re-ranker.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        if rerank_score != 0:
            # Cross-encoder scores are typically in [-10, 10] range
            # Sigmoid normalization to [0, 1]
            import math
            normalized_rerank = 1 / (1 + math.exp(-rerank_score))
            # Normalize retrieval (RRF scores are typically small)
            normalized_retrieval = min(retrieval_score * 30, 1.0)  # Scale up RRF
            # Weighted average
            confidence = 0.3 * normalized_retrieval + 0.7 * normalized_rerank
        else:
            # No re-ranker — use retrieval score only
            confidence = min(retrieval_score * 30, 1.0)

        return max(0.0, min(1.0, confidence))

    def format_answer(
        self,
        sentences: List[str],
        source: str,
        page: Optional[int],
        confidence: float,
    ) -> Answer:
        """
        Format raw components into a structured Answer.

        Args:
            sentences: Answer sentences.
            source: Document source.
            page: Page number.
            confidence: Confidence score.

        Returns:
            Answer object.
        """
        text = " ".join(sentences)
        entities = self._ner.get_answer_entities(text, "OTHER")

        return Answer(
            text=text,
            source=source,
            page=page,
            confidence=confidence,
            supporting_passage=text,
            entities=entities,
        )

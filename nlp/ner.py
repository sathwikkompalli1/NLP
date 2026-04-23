"""
IQAS Named Entity Recognition
===============================
Extract, filter, and highlight named entities using spaCy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from nlp.tokenizer import NLPTokenizer
from utils.logger import get_logger

log = get_logger("ner")


# ──────────────────────────── Data Model ────────────────────────────


@dataclass
class Entity:
    """A recognized named entity with span information."""
    text: str
    label: str
    start: int
    end: int


# ──────────────────────────── Entity Colors ────────────────────────────

ENTITY_COLORS = {
    "PERSON": "#FF6B6B",
    "ORG": "#4ECDC4",
    "GPE": "#45B7D1",
    "DATE": "#96CEB4",
    "TIME": "#96CEB4",
    "MONEY": "#FFEAA7",
    "PERCENT": "#DFE6E9",
    "LOC": "#45B7D1",
    "EVENT": "#FD79A8",
    "PRODUCT": "#A29BFE",
    "WORK_OF_ART": "#E17055",
    "LAW": "#FDCB6E",
    "NORP": "#6C5CE7",
    "FAC": "#00B894",
    "LANGUAGE": "#E84393",
    "CARDINAL": "#B2BEC3",
    "ORDINAL": "#B2BEC3",
    "QUANTITY": "#B2BEC3",
}


# ──────────────────────────── NER Extractor ────────────────────────────


class NERExtractor:
    """
    Named Entity Recognition using spaCy.

    Extracts, filters, and visualizes named entities in text.
    """

    def __init__(self):
        """Initialize with shared spaCy model."""
        self._tokenizer = NLPTokenizer()
        self.nlp = self._tokenizer.nlp

    def extract(self, text: str) -> List[Entity]:
        """
        Extract all named entities from text.

        Args:
            text: Input text.

        Returns:
            List of Entity objects with text, label, start, end.
        """
        doc = self.nlp(text)
        return [
            Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            )
            for ent in doc.ents
        ]

    def get_entities_by_type(self, text: str, label: str) -> List[str]:
        """
        Extract entities of a specific type.

        Args:
            text: Input text.
            label: Entity label to filter (e.g., "PERSON", "ORG", "DATE").

        Returns:
            List of entity text strings matching the label.
        """
        entities = self.extract(text)
        return list(set(ent.text for ent in entities if ent.label == label.upper()))

    def get_all_entity_types(self, text: str) -> dict[str, List[str]]:
        """
        Group all entities by their type.

        Returns:
            Dict mapping entity label → list of entity texts.
        """
        entities = self.extract(text)
        grouped: dict[str, List[str]] = {}
        for ent in entities:
            if ent.label not in grouped:
                grouped[ent.label] = []
            if ent.text not in grouped[ent.label]:
                grouped[ent.label].append(ent.text)
        return grouped

    def highlight_entities(self, text: str) -> str:
        """
        Return HTML-formatted text with highlighted entities.

        Each entity is wrapped in a colored <span> tag based on its type.

        Args:
            text: Input text.

        Returns:
            HTML string with highlighted entities.
        """
        entities = self.extract(text)
        if not entities:
            return text

        # Sort by start position (reverse) to avoid offset issues
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        result = text
        for ent in sorted_entities:
            color = ENTITY_COLORS.get(ent.label, "#B2BEC3")
            badge = (
                f'<span style="background-color: {color}; color: #1a1a2e; '
                f'padding: 2px 6px; border-radius: 4px; font-weight: 600; '
                f'font-size: 0.9em; margin: 0 2px;">'
                f'{ent.text}'
                f'<span style="font-size: 0.7em; opacity: 0.8; margin-left: 4px;">'
                f'{ent.label}</span></span>'
            )
            result = result[:ent.start] + badge + result[ent.end:]

        return result

    def get_answer_entities(self, text: str, q_type: str) -> List[str]:
        """
        Extract entities relevant to a question type.

        Priority mapping:
            WHO   → PERSON, ORG
            WHEN  → DATE, TIME
            WHERE → GPE, LOC, FAC
            WHAT  → all entities
            DEFINE → all entities

        Args:
            text: Text to extract entities from.
            q_type: Question type (WHO/WHAT/WHEN/WHERE/WHY/HOW/DEFINE).

        Returns:
            List of relevant entity texts.
        """
        priority_map = {
            "WHO": ["PERSON", "ORG", "NORP"],
            "WHEN": ["DATE", "TIME", "EVENT"],
            "WHERE": ["GPE", "LOC", "FAC"],
            "WHAT": None,   # All types
            "DEFINE": None,
            "HOW": None,
            "WHY": None,
            "OTHER": None,
        }

        target_labels = priority_map.get(q_type.upper())
        entities = self.extract(text)

        if target_labels is None:
            # Return all unique entity texts
            return list(set(ent.text for ent in entities))

        # Filter and prioritize
        relevant = [ent.text for ent in entities if ent.label in target_labels]
        return list(set(relevant)) if relevant else list(set(ent.text for ent in entities))

    def get_entity_count(self, text: str) -> dict[str, int]:
        """
        Count entities by type.

        Returns:
            Dict mapping entity label → count.
        """
        entities = self.extract(text)
        counts: dict[str, int] = {}
        for ent in entities:
            counts[ent.label] = counts.get(ent.label, 0) + 1
        return counts

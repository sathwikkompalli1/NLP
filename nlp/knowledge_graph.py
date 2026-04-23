"""
IQAS Knowledge Graph Builder
==============================
Automatic entity-relationship extraction using spaCy dependency parsing + NER.
Builds interactive graph data with nodes, edges, and co-occurrence matrices.

NLP Techniques:
    - Named Entity Recognition (spaCy)
    - Dependency Parsing for triple extraction (Subject → Verb → Object)
    - Noun Phrase extraction for richer nodes
    - Sentence segmentation for co-occurrence analysis
    - POS tagging for relationship verb identification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from nlp.tokenizer import NLPTokenizer
from nlp.ner import NERExtractor, Entity
from utils.logger import get_logger

log = get_logger("knowledge_graph")


# ──────────────────────────── Data Models ────────────────────────────


@dataclass
class Triple:
    """A Subject → Relation → Object triple extracted from text."""
    subject: str
    relation: str
    object: str
    sentence: str      # Source sentence
    confidence: float  # Extraction confidence


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    id: str
    label: str
    entity_type: str   # PERSON, ORG, GPE, CONCEPT, etc.
    frequency: int
    color: str


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""
    source: str
    target: str
    relation: str
    weight: float


@dataclass
class KnowledgeGraphData:
    """Complete knowledge graph data for visualization."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    triples: List[Triple]
    entity_frequencies: Dict[str, int]
    cooccurrence_matrix: Dict[str, Dict[str, int]]
    entity_types: Dict[str, str]


# ──────────────────────────── Entity Colors ────────────────────────────

ENTITY_TYPE_COLORS = {
    "PERSON": "#FF6B6B",
    "ORG": "#4ECDC4",
    "GPE": "#45B7D1",
    "LOC": "#45B7D1",
    "DATE": "#96CEB4",
    "EVENT": "#FD79A8",
    "PRODUCT": "#A29BFE",
    "WORK_OF_ART": "#E17055",
    "NORP": "#6C5CE7",
    "FAC": "#00B894",
    "CONCEPT": "#00B4D8",
    "OTHER": "#B2BEC3",
}


# ──────────────────────────── Knowledge Graph Builder ────────────────────────────


class KnowledgeGraphBuilder:
    """
    Build knowledge graphs from text using NLP.

    Extracts:
        - Named entities as nodes
        - Subject-Verb-Object triples as edges
        - Entity co-occurrence relationships
        - Noun phrases as concept nodes
    """

    def __init__(self):
        """Initialize with shared spaCy model."""
        self._tokenizer = NLPTokenizer()
        self._ner = NERExtractor()
        self.nlp = self._tokenizer.nlp

    def build_graph(self, text: str) -> KnowledgeGraphData:
        """
        Build a complete knowledge graph from text.

        Args:
            text: Input text.

        Returns:
            KnowledgeGraphData with nodes, edges, triples, and co-occurrence.
        """
        # Extract triples
        triples = self.extract_triples(text)

        # Extract entities
        entities = self._ner.extract(text)
        entity_freq = self._count_entities(entities)
        entity_types = {ent.text: ent.label for ent in entities}

        # Build co-occurrence matrix
        cooccurrence = self.get_entity_cooccurrence(text)

        # Build nodes from entities + triple subjects/objects
        nodes = self._build_nodes(entity_freq, entity_types, triples)

        # Build edges from triples + co-occurrence
        edges = self._build_edges(triples, cooccurrence)

        return KnowledgeGraphData(
            nodes=nodes,
            edges=edges,
            triples=triples,
            entity_frequencies=entity_freq,
            cooccurrence_matrix=cooccurrence,
            entity_types=entity_types,
        )

    def extract_triples(self, text: str) -> List[Triple]:
        """
        Extract Subject → Relation → Object triples using dependency parsing.

        Process:
            1. Parse each sentence with spaCy
            2. Find ROOT verb
            3. Find subject (nsubj/nsubjpass) and object (dobj/attr/pobj)
            4. Expand subjects/objects to full noun phrases

        Args:
            text: Input text.

        Returns:
            List of Triple objects.
        """
        doc = self.nlp(text)
        triples = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Find the root verb(s)
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                    subject = None
                    obj = None

                    # Find subject
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            subject = self._expand_noun_phrase(child)
                        elif child.dep_ in ("dobj", "attr", "oprd"):
                            obj = self._expand_noun_phrase(child)

                    # Try to find object in prepositional phrases
                    if obj is None:
                        for child in token.children:
                            if child.dep_ == "prep":
                                for grandchild in child.children:
                                    if grandchild.dep_ == "pobj":
                                        obj = self._expand_noun_phrase(grandchild)
                                        break
                                if obj:
                                    break

                    # If we found both subject and object
                    if subject and obj and subject != obj:
                        # Skip very short or generic terms
                        if len(subject) > 1 and len(obj) > 1:
                            triples.append(Triple(
                                subject=subject,
                                relation=token.lemma_.lower(),
                                object=obj,
                                sentence=sent_text,
                                confidence=self._compute_triple_confidence(token, subject, obj),
                            ))

                # Also handle nominal subjects with copula
                elif token.dep_ == "ROOT" and token.pos_ in ("NOUN", "ADJ"):
                    subject = None
                    for child in token.children:
                        if child.dep_ == "nsubj":
                            subject = self._expand_noun_phrase(child)

                    if subject:
                        obj = self._expand_noun_phrase(token)
                        if subject != obj and len(subject) > 1 and len(obj) > 1:
                            triples.append(Triple(
                                subject=subject,
                                relation="is",
                                object=obj,
                                sentence=sent_text,
                                confidence=0.6,
                            ))

        # Deduplicate
        seen = set()
        unique_triples = []
        for t in triples:
            key = (t.subject.lower(), t.relation.lower(), t.object.lower())
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)

        return unique_triples

    def get_entity_cooccurrence(self, text: str) -> Dict[str, Dict[str, int]]:
        """
        Build entity co-occurrence matrix (entities appearing in the same sentence).

        Returns:
            Nested dict: entity_a → entity_b → count.
        """
        sentences = self._tokenizer.sent_tokenize(text)
        cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for sent in sentences:
            entities = self._ner.extract(sent)
            unique_entities = list(set(ent.text for ent in entities))

            # Pairwise co-occurrence
            for i in range(len(unique_entities)):
                for j in range(i + 1, len(unique_entities)):
                    a, b = unique_entities[i], unique_entities[j]
                    cooccurrence[a][b] += 1
                    cooccurrence[b][a] += 1

        return dict(cooccurrence)

    def get_graph_data(self, text: str) -> Dict:
        """
        Get graph data in a format suitable for plotly visualization.

        Returns:
            Dict with 'nodes' and 'edges' lists ready for plotting.
        """
        graph = self.build_graph(text)

        nodes_data = [
            {
                "id": n.id,
                "label": n.label,
                "type": n.entity_type,
                "frequency": n.frequency,
                "color": n.color,
            }
            for n in graph.nodes
        ]

        edges_data = [
            {
                "source": e.source,
                "target": e.target,
                "relation": e.relation,
                "weight": e.weight,
            }
            for e in graph.edges
        ]

        return {"nodes": nodes_data, "edges": edges_data}

    # ──────────────────── Private Helpers ────────────────────

    def _expand_noun_phrase(self, token) -> str:
        """Expand a token to its full noun phrase using the subtree."""
        # Get the noun chunk if available
        if token.doc[token.i:token.i + 1].as_doc().text:
            # Collect relevant subtree tokens
            subtree_tokens = []
            for t in token.subtree:
                if t.dep_ in ("compound", "amod", "nmod", "poss") or t == token:
                    subtree_tokens.append(t)
                elif t.dep_ == "det" and t.i < token.i:
                    continue  # Skip determiners

            if subtree_tokens:
                subtree_tokens.sort(key=lambda t: t.i)
                phrase = " ".join(t.text for t in subtree_tokens)
                return phrase.strip()

        return token.text

    def _count_entities(self, entities: List[Entity]) -> Dict[str, int]:
        """Count entity frequency."""
        freq: Dict[str, int] = {}
        for ent in entities:
            freq[ent.text] = freq.get(ent.text, 0) + 1
        return freq

    def _build_nodes(
        self,
        entity_freq: Dict[str, int],
        entity_types: Dict[str, str],
        triples: List[Triple],
    ) -> List[GraphNode]:
        """Build graph nodes from entities and triple participants."""
        all_nodes: Dict[str, GraphNode] = {}

        # Add entity nodes
        for entity, count in entity_freq.items():
            etype = entity_types.get(entity, "OTHER")
            color = ENTITY_TYPE_COLORS.get(etype, ENTITY_TYPE_COLORS["OTHER"])
            node_id = entity.lower().replace(" ", "_")
            all_nodes[node_id] = GraphNode(
                id=node_id,
                label=entity,
                entity_type=etype,
                frequency=count,
                color=color,
            )

        # Add triple participants as concept nodes if not already present
        for triple in triples:
            for term in [triple.subject, triple.object]:
                node_id = term.lower().replace(" ", "_")
                if node_id not in all_nodes:
                    all_nodes[node_id] = GraphNode(
                        id=node_id,
                        label=term,
                        entity_type="CONCEPT",
                        frequency=1,
                        color=ENTITY_TYPE_COLORS["CONCEPT"],
                    )

        return list(all_nodes.values())

    def _build_edges(
        self,
        triples: List[Triple],
        cooccurrence: Dict[str, Dict[str, int]],
    ) -> List[GraphEdge]:
        """Build graph edges from triples and co-occurrence."""
        edges: Dict[str, GraphEdge] = {}

        # Add triple edges
        for triple in triples:
            src_id = triple.subject.lower().replace(" ", "_")
            tgt_id = triple.object.lower().replace(" ", "_")
            edge_key = f"{src_id}→{tgt_id}"

            if edge_key not in edges:
                edges[edge_key] = GraphEdge(
                    source=src_id,
                    target=tgt_id,
                    relation=triple.relation,
                    weight=triple.confidence,
                )
            else:
                # Increase weight for repeated relations
                edges[edge_key].weight = min(edges[edge_key].weight + 0.2, 1.0)

        # Add co-occurrence edges (weaker)
        for entity_a, neighbors in cooccurrence.items():
            for entity_b, count in neighbors.items():
                src_id = entity_a.lower().replace(" ", "_")
                tgt_id = entity_b.lower().replace(" ", "_")
                edge_key = f"{src_id}→{tgt_id}"
                rev_key = f"{tgt_id}→{src_id}"

                if edge_key not in edges and rev_key not in edges:
                    edges[edge_key] = GraphEdge(
                        source=src_id,
                        target=tgt_id,
                        relation="co-occurs with",
                        weight=min(count * 0.3, 1.0),
                    )

        return list(edges.values())

    def _compute_triple_confidence(self, verb_token, subject: str, obj: str) -> float:
        """Compute confidence score for a triple based on parse quality."""
        confidence = 0.5

        # Boost if verb is a proper verb (not auxiliary)
        if verb_token.pos_ == "VERB":
            confidence += 0.15

        # Boost if subject/object are multi-word (more specific)
        if " " in subject:
            confidence += 0.1
        if " " in obj:
            confidence += 0.1

        # Boost if subject is a named entity
        doc = verb_token.doc
        for ent in doc.ents:
            if subject.lower() in ent.text.lower():
                confidence += 0.1
                break

        return min(confidence, 1.0)

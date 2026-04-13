"""
Core data models for the four-layer architecture.

This module defines all the core data structures used across all layers:
- Document types and their configurations
- Concept types and relationship types
- Dataclasses for concepts, relationships, and concept graphs
- Output structures for each layer
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class DocumentType(Enum):
    """Document type for optimized extraction strategy"""
    LECTURE = "lecture"        # Lecture slides / course materials
    PAPER = "paper"            # Academic paper
    BOOK = "book"              # Book (future support)
    AUTO = "auto"              # Auto-detect


class RelationshipType(Enum):
    """Predefined relationship types between concepts"""
    # Core relationships
    PREREQUISITE = "prerequisite"      # A is prerequisite for B
    DERIVATION = "derivation"          # B is derived from A
    APPLICATION = "application"        # A is applied in B
    COMPARISON = "comparison"          # A vs B comparison
    GENERALIZATION = "generalization"  # A is generalization of B
    SPECIALIZATION = "specialization"  # B is specialization of A
    COMPONENT = "component"            # A is component of B

    # Academic paper specific relationships
    SOLVES = "solves"                  # MODEL -> PROBLEM
    EVALUATED_ON = "evaluated_on"      # MODEL -> DATASET
    IMPROVES_UPON = "improves_upon"    # MODEL -> BASELINE
    USES = "uses"                      # FRAMEWORK -> ALGORITHM
    CONTRADICTION = "contradiction"    # Bidirectional conflict
    SYNONYM = "synonym"                # Bidirectional synonym/abbreviation

    # AI identified special cases
    AI_IDENTIFIED = "ai_identified"


class ConceptType(Enum):
    """What the concept itself is (for application layer generation)"""
    # General types
    THEOREM = "theorem"
    DEFINITION = "definition"
    ALGORITHM = "algorithm"
    MODEL = "model"
    COMMAND = "command"
    METRIC = "metric"
    FRAMEWORK = "framework"
    DATASET = "dataset"
    TECHNIQUE = "technique"

    # Paper specific types
    PROBLEM = "problem"           # Problem / Challenge
    HYPOTHESIS = "hypothesis"     # Hypothesis / Assumption
    FINDING = "finding"           # Finding / Conclusion
    EXPERIMENT = "experiment"     # Evaluation / Experiment
    HARDWARE = "hardware"         # Hardware / Environment

    UNKNOWN = "unknown"


class PaperElement(Enum):
    """Role of a concept in the paper"""
    PROPOSED_METHOD = "proposed_method"    # Method proposed in the paper
    BASELINE = "baseline"                  # Comparison baseline
    BACKGROUND = "background"              # Background knowledge
    MOTIVATION = "motivation"              # Motivation / Problem
    EVALUATION = "evaluation"              # Evaluation content
    FUTURE_WORK = "future_work"            # Future work
    UNKNOWN = "unknown"


# Document type configuration
DOCUMENT_TYPE_CONFIG = {
    DocumentType.LECTURE: {
        "name": "课件/讲义",
        "description": "教学课件，侧重概念讲解和示例",
        "relevant_concept_types": [
            ConceptType.DEFINITION,
            ConceptType.THEOREM,
            ConceptType.ALGORITHM,
            ConceptType.MODEL,
            ConceptType.COMMAND,
            ConceptType.TECHNIQUE,
        ],
        "relevant_paper_elements": [
            PaperElement.BACKGROUND,
        ],
        "relevant_relationship_types": [
            RelationshipType.PREREQUISITE,
            RelationshipType.DERIVATION,
            RelationshipType.APPLICATION,
            RelationshipType.COMPARISON,
            RelationshipType.SYNONYM,
        ],
        "sections": ["简介", "概念", "定义", "例子", "练习"],
    },
    DocumentType.PAPER: {
        "name": "学术论文",
        "description": "研究论文，包含完整的研究流程",
        "relevant_concept_types": [
            # General concepts
            ConceptType.DEFINITION,
            ConceptType.THEOREM,
            ConceptType.ALGORITHM,
            ConceptType.MODEL,
            ConceptType.TECHNIQUE,
            ConceptType.METRIC,
            ConceptType.DATASET,
            ConceptType.FRAMEWORK,
            ConceptType.HARDWARE,
            # Paper specific concepts
            ConceptType.PROBLEM,
            ConceptType.HYPOTHESIS,
            ConceptType.FINDING,
            ConceptType.EXPERIMENT,
        ],
        "relevant_paper_elements": [
            PaperElement.PROPOSED_METHOD,
            PaperElement.BASELINE,
            PaperElement.BACKGROUND,
            PaperElement.MOTIVATION,
            PaperElement.EVALUATION,
            PaperElement.FUTURE_WORK,
        ],
        "relevant_relationship_types": [
            RelationshipType.PREREQUISITE,
            RelationshipType.DERIVATION,
            RelationshipType.APPLICATION,
            RelationshipType.COMPARISON,
            RelationshipType.SOLVES,
            RelationshipType.IMPROVES_UPON,
            RelationshipType.EVALUATED_ON,
            RelationshipType.USES,
            RelationshipType.CONTRADICTION,
            RelationshipType.SYNONYM,
        ],
        "sections": ["Abstract", "Introduction", "Method", "Experiment", "Conclusion", "Related Work"],
    },
}


# Bidirectional relationship types
BIDIRECTIONAL_RELATIONSHIPS = {
    RelationshipType.COMPARISON,
    RelationshipType.SYNONYM,
    RelationshipType.CONTRADICTION,
}


@dataclass
class Concept:
    """Represents a single concept extracted from PDF"""
    name: str
    introduction: str
    context: str

    # Extended fields
    aliases: List[str] = field(default_factory=list)     # Synonyms / abbreviations
    section_name: Optional[str] = None                   # Section name (Abstract/Method/etc)
    importance_score: int = 1                            # Importance score 1-5 (AI dynamic)
    paper_role: PaperElement = PaperElement.UNKNOWN      # Role in paper

    # Original fields
    page_number: Optional[int] = None
    concept_type: ConceptType = ConceptType.UNKNOWN


@dataclass
class Relationship:
    """Represents a relationship between two concepts"""
    source_concept: str
    target_concept: str
    relationship_type: RelationshipType
    description: str

    # Extended fields
    evidence_quote: str = ""                             # Original quote evidence (reduces hallucination)
    is_bidirectional: bool = False                       # Whether bidirectional

    # Original fields
    confidence: float = 1.0


@dataclass
class ConceptGraph:
    """Graph of concepts and their relationships"""
    concepts: Dict[str, Concept] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    source_document: str = ""

    # Extended fields
    metadata: Dict[str, str] = field(default_factory=dict)  # Paper metadata (title, author, date)
    global_summary: str = ""                                # Global summary based on graph

    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the graph"""
        self.concepts[concept.name] = concept

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph"""
        self.relationships.append(relationship)

    def get_related_concepts(self, concept_name: str) -> List[str]:
        """Get all concepts related to the given concept"""
        related = set()
        for rel in self.relationships:
            if rel.source_concept == concept_name:
                related.add(rel.target_concept)
            elif rel.target_concept == concept_name:
                related.add(rel.source_concept)
        return list(related)

    def get_relationships_between(
        self, concept1: str, concept2: str
    ) -> List[Relationship]:
        """Get all relationships between two concepts"""
        return [
            rel for rel in self.relationships
            if (rel.source_concept == concept1 and rel.target_concept == concept2) or
               (rel.source_concept == concept2 and rel.target_concept == concept1)
        ]


# Layer output structures
@dataclass
class Layer1Output:
    """Output from Concept Extraction Layer"""
    concept_graph: ConceptGraph
    processing_time: float = 0.0


@dataclass
class Layer2Output:
    """Output from Detail Generation Layer"""
    detail_contents: Dict[str, str]                # concept -> detail content
    relationship_contents: Dict[str, str]          # relationship -> content
    sources_used: Dict[str, List[str]]             # concept -> source URLs


@dataclass
class Layer3Output:
    """Output from Application Generation Layer"""
    application_contents: Dict[str, str]           # concept -> application content
    concept_types: Dict[str, ConceptType]          # concept -> classified type
    code_examples: Dict[str, List[str]]           # concept -> code examples
    industrial_cases: Dict[str, List[str]]        # concept -> industrial use cases


@dataclass
class FourLayerSummary:
    """Complete four-layer summary output"""
    layer1: Layer1Output
    layer2: Layer2Output
    layer3: Layer3Output
    metadata: Dict[str, str] = field(default_factory=dict)

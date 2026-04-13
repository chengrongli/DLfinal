"""
Detail generation layer for the four-layer architecture.

This module generates detailed content based on relationship types.
"""

from __future__ import annotations

from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from four_layer_agent.src.core.config import Settings
    from four_layer_agent.src.core.data_models import (
        Layer1Output, Layer2Output,
        Concept, ConceptGraph, Relationship
    )

from four_layer_agent.src.layer_2_detail.relationship_content import RelationshipContent
from four_layer_agent.src.core.data_models import Layer1Output, Layer2Output


class DetailGenerator:
    """Generate detailed content based on relationships."""

    def __init__(self, settings, llm_client):
        """
        Initialize detail generator.

        Args:
            settings: Settings instance
            llm_client: LLM client for text generation
        """
        self.settings = settings
        self.llm = llm_client
        self.relationship_content = RelationshipContent(settings, llm_client)

    def process(self, layer1: Layer1Output) -> Layer2Output:
        """
        Generate detail content for concepts and relationships.

        Args:
            layer1: Output from concept extraction layer

        Returns:
            Layer2Output with detailed contents
        """
        detail_contents = {}
        relationship_contents = {}
        sources_used = {}

        graph = layer1.concept_graph
        domain = graph.metadata.get("domain", "")

        for concept_name, concept in graph.concepts.items():
            # Get relationships for this concept
            relationships = self._get_concept_relationships(concept_name, graph)

            # Generate detail content
            detail = self._generate_concept_detail(concept, relationships, domain)
            detail_contents[concept_name] = detail

            # Generate content for each relationship
            for rel in relationships:
                # Merged cluster: source_concept is "A、B、C", target is empty
                if rel.target_concept:
                    rel_key = f"{rel.source_concept}->{rel.target_concept}"
                else:
                    rel_key = rel.source_concept  # e.g. "能量基模型、变分自编码器、对抗生成网络"
                if rel_key not in relationship_contents:
                    relationship_contents[rel_key] = self.relationship_content.generate(
                        rel, graph
                    )

        return Layer2Output(
            detail_contents=detail_contents,
            relationship_contents=relationship_contents,
            sources_used=sources_used
        )

    def _get_concept_relationships(
        self, concept_name: str, graph: ConceptGraph
    ) -> List[Relationship]:
        """Get all relationships involving this concept."""
        return [
            rel for rel in graph.relationships
            if concept_name in rel.source_concept or rel.target_concept == concept_name
        ]

    def _generate_concept_detail(
        self, concept: Concept, relationships: List[Relationship],
        domain: str = ""
    ) -> str:
        """Generate detailed explanation of a concept."""
        domain_hint = ""
        if domain:
            domain_hint = f"\n注意：请从{domain}领域的角度解释该概念，而非其原始学科领域。"

        prompt = f"""请详细解释以下概念：

概念名称：{concept.name}
介绍：{concept.introduction}
重要性：{concept.importance_score}/5{domain_hint}

请提供：
1. 概念的详细定义
2. 核心原理或机制
3. 关键特点

输出简体中文，300字以内，确保内容完整。不要加前缀套话，直接输出内容。
"""

        try:
            return self.llm.generate(prompt, max_new_tokens=512, max_chars=300).strip()
        except Exception:
            return f"{concept.name}：{concept.introduction}"

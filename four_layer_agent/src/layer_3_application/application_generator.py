"""
Application generation layer for the four-layer architecture.

This module generates application examples based on concept types.
"""

from __future__ import annotations

import re
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from four_layer_agent.src.core.config import Settings
    from four_layer_agent.src.core.data_models import (
        Layer1Output, Layer2Output, Layer3Output,
        Concept, ConceptType
    )

from four_layer_agent.src.layer_3_application.concept_classifier import ConceptClassifier
from four_layer_agent.src.layer_3_application.type_content import TypeContent
from four_layer_agent.src.core.data_models import Layer1Output, Layer2Output, Layer3Output, ConceptType


class ApplicationGenerator:
    """Generate application examples based on concept types."""

    def __init__(self, settings, llm_client):
        """
        Initialize application generator.

        Args:
            settings: Settings instance
            llm_client: LLM client for text generation
        """
        self.settings = settings
        self.llm = llm_client
        self.concept_classifier = ConceptClassifier(settings, llm_client)
        self.type_content = TypeContent(settings, llm_client)

    def process(self, layer1: Layer1Output, layer2: Layer2Output) -> Layer3Output:
        """
        Generate application content for each concept.

        Args:
            layer1: Output from concept extraction layer
            layer2: Output from detail generation layer

        Returns:
            Layer3Output with application contents
        """
        application_contents = {}
        concept_types = {}
        code_examples = {}
        industrial_cases = {}

        graph = layer1.concept_graph
        detail_contents = layer2.detail_contents
        domain = graph.metadata.get("domain", "")

        for concept_name, concept in graph.concepts.items():
            # Classify concept type
            concept_type = self.concept_classifier.classify(
                concept, detail_contents.get(concept_name, "")
            )
            concept_types[concept_name] = concept_type

            # Generate application content
            app_content = self.type_content.generate_application(
                concept, concept_type, detail_contents.get(concept_name, ""),
                domain=domain
            )
            application_contents[concept_name] = app_content

            # Extract type-specific examples
            if concept_type == ConceptType.COMMAND:
                code_examples[concept_name] = self._extract_code_examples(app_content)
            elif concept_type == ConceptType.MODEL:
                industrial_cases[concept_name] = self._extract_industrial_cases(app_content)

        return Layer3Output(
            application_contents=application_contents,
            concept_types=concept_types,
            code_examples=code_examples,
            industrial_cases=industrial_cases
        )

    def _extract_code_examples(self, content: str) -> List[str]:
        """Extract code examples from content."""
        examples = []

        # Extract code blocks
        code_blocks = re.findall(r'```(?:bash|python|sh)?\n(.*?)```', content, re.DOTALL)
        examples.extend(code_blocks)

        # Extract inline code patterns
        inline_codes = re.findall(r'`([^`\n]+)`', content)
        examples.extend(inline_codes)

        return examples[:5]  # Limit to 5 examples

    def _extract_industrial_cases(self, content: str) -> List[str]:
        """Extract industrial use cases from content."""
        cases = []

        # Split by common delimiters
        parts = re.split(r'[。；;]', content)
        for part in parts:
            # Look for company/product mentions
            if any(keyword in part for keyword in ['公司', '企业', '产品', '应用', 'Google', 'OpenAI', '微软']):
                cases.append(part.strip())

        return cases[:5]  # Limit to 5 cases

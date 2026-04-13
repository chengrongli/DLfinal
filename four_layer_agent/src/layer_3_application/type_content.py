"""
Type content generator for the four-layer architecture.

This module generates application content based on concept types.
"""

from typing import TYPE_CHECKING

from four_layer_agent.src.core.data_models import Concept, ConceptType

if TYPE_CHECKING:
    from four_layer_agent.src.core.config import Settings


class TypeContent:
    """Generate application content based on concept types."""

    TEMPLATES = {
        "definition": """为概念"{concept}"提供实际应用说明。
包括：
- 适用场景和实际用途
- 相关的工具、库或产品
- 应用效果或性能表现
""",
    }

    def __init__(self, settings, llm_client):
        """
        Initialize type content generator.

        Args:
            settings: Settings instance
            llm_client: LLM client for text generation
        """
        self.settings = settings
        self.llm = llm_client

    def generate_application(
        self, concept, concept_type: ConceptType, detail: str,
        domain: str = ""
    ) -> str:
        """
        Generate application content based on concept type.

        Args:
            concept: Concept object
            concept_type: Classified concept type
            detail: Detail content from Layer 3
            domain: Paper's domain context (e.g., "machine learning")

        Returns:
            Generated application content
        """
        template = self.TEMPLATES.get(
            concept_type.value,
            "为{concept}提供应用示例"
        )

        prompt = template.format(concept=concept.name)

        # Add section note if available
        section_note = ""
        if concept.section_name:
            section_note = f"\n（该概念出现在 {concept.section_name} 章节）"

        domain_hint = ""
        if domain:
            domain_hint = f"\n注意：请从{domain}领域的角度描述应用场景。"

        context = f"""介绍：{concept.introduction}{section_note}{domain_hint}
重要性评分：{concept.importance_score}/5
技术细节：{detail[:1000]}

生成应用内容（简体中文，200字以内，确保内容完整，不要加前缀套话）：
"""

        full_prompt = prompt + "\n" + context

        try:
            return self.llm.generate(full_prompt, max_new_tokens=512, max_chars=200).strip()
        except Exception:
            return f"{concept.name}的应用场景包括实际项目和学术研究中的多个领域。"

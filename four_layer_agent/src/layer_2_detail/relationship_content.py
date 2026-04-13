"""
Relationship content generator for the four-layer architecture.

This module generates content based on relationship types.
"""

from typing import Dict, List, TYPE_CHECKING

from four_layer_agent.src.core.data_models import (
    Concept, ConceptGraph, Relationship, RelationshipType
)

if TYPE_CHECKING:
    from four_layer_agent.src.core.config import Settings


class RelationshipContent:
    """Generate content based on relationship types."""

    TEMPLATES = {
        # Core relationships
        "derivation": """解释如何从{source}推导得到{target}。
包括：数学推导步骤、逻辑推理过程、关键定理引用。
""",

        "prerequisite": """解释为什么{source}是理解{target}的前置条件。
包括：依赖关系、学习路径、基础概念说明。
""",

        "comparison": """对比{source}和{target}的异同。
包括：相似点、不同点、适用场景对比。
""",

        "component": """说明{source}是{target}的组成部分。
解释{source}在整个{target}中的作用。
""",

        # Academic paper specific relationships
        "solves": """详细解释{source}如何解决{target}问题。
包括：
- 问题的具体描述和痛点
- {source}的解决方案核心思路
- 为什么{source}能解决这个问题
""",

        "improves_upon": """详细分析{source}相比{target}的改进。
包括：
- {target}的局限性分析
- {source}的具体改进点
- 性能提升的对比
""",

        "evaluated_on": """分析{source}在{target}数据集上的评估。
包括：
- 数据集特点介绍
- 评估指标和方法
- 主要实验结果
""",

        "uses": """解释{source}如何使用{target}。
包括：
- 使用目的和动机
- 具体实现方式
""",

        "synonym": """说明{source}和{target}是同义词/缩写关系。
在后续内容中将统一使用主名称：{source}
""",
    }

    def __init__(self, settings, llm_client):
        """
        Initialize relationship content generator.

        Args:
            settings: Settings instance
            llm_client: LLM client for text generation
        """
        self.settings = settings
        self.llm = llm_client

    def generate(
        self, rel: Relationship, graph: ConceptGraph
    ) -> str:
        """
        Generate content for a specific relationship type.

        Args:
            rel: Relationship object
            graph: Concept graph containing all concepts

        Returns:
            Generated content about the relationship
        """
        # Get template for this relationship type
        template = self.TEMPLATES.get(
            rel.relationship_type.value,
            "解释{source}和{target}之间的关系：{description}"
        )

        source_concept = graph.concepts.get(rel.source_concept)
        target_concept = graph.concepts.get(rel.target_concept) if rel.target_concept else None

        # Merged cluster comparison: source_concept is "A、B、C", no target
        is_merged_cluster = rel.relationship_type.value == "comparison" and not rel.target_concept

        if is_merged_cluster:
            prompt = f"对比以下概念的异同（包括相似点、不同点、适用场景）：{rel.description}"
        elif rel.relationship_type.value == "comparison" and rel.description:
            prompt = f"对比以下概念的异同（包括相似点、不同点、适用场景）：{rel.description}"
        else:
            prompt = template.format(
                source=rel.source_concept,
                target=rel.target_concept,
                description=rel.description
            )

        # Add evidence if available
        evidence = f"\n原句证据：{rel.evidence_quote}" if rel.evidence_quote else ""

        # Add concept information — include all concepts mentioned in description
        # for merged cluster comparisons (e.g. "A、B、C 为同类概念")
        concept_intros = []
        for name, concept in graph.concepts.items():
            if name in rel.description or name in rel.source_concept or name == rel.target_concept:
                concept_intros.append(f"- {name}：{concept.introduction}")
        context = "概念列表：\n" + "\n".join(concept_intros) + evidence

        full_prompt = prompt + "\n" + context + "\n请生成详细说明（简体中文，300字以内，确保内容完整，不要加前缀套话）："

        try:
            return self.llm.generate(full_prompt, max_new_tokens=512, max_chars=300).strip()
        except Exception:
            return f"{rel.source_concept}与{rel.target_concept}存在{rel.relationship_type.value}关系。"

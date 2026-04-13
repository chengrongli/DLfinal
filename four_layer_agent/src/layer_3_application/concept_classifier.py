"""
Concept classifier for the four-layer architecture.

This module uses AI to classify concepts into types.
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from four_layer_agent.src.core.config import Settings
    from four_layer_agent.src.core.data_models import Concept, ConceptType

from four_layer_agent.src.core.data_models import ConceptType


class ConceptClassifier:
    """Classify concepts into types using AI inference."""

    def __init__(self, settings, llm_client):
        """
        Initialize concept classifier.

        Args:
            settings: Settings instance
            llm_client: LLM client for text generation
        """
        self.settings = settings
        self.llm = llm_client

    def classify(self, concept, detail_content: str = "") -> ConceptType:
        """
        Classify a concept into its type.

        Returns ConceptType.DEFINITION for all concepts (simplified).
        """
        return ConceptType.DEFINITION

    def _classify_by_keywords(self, concept) -> ConceptType:
        """Classify concept based on keywords in name and introduction."""
        name_lower = concept.name.lower()
        intro_lower = concept.introduction.lower()

        # Check for command patterns
        if any(cmd in name_lower for cmd in ['grep', 'ls', 'cd', 'python', 'git', 'docker']):
            return ConceptType.COMMAND

        # Check for framework patterns
        if any(fw in name_lower for fw in ['pytorch', 'tensorflow', 'keras', 'scikit', 'numpy']):
            return ConceptType.FRAMEWORK

        # Check for model patterns
        if any(m in name_lower for m in ['gan', 'transformer', 'bert', 'gpt', 'resnet', 'vae']):
            return ConceptType.MODEL

        # Check for algorithm patterns
        if any(a in name_lower for a in ['算法', 'algorithm', 'sort', 'search', 'optimization']):
            return ConceptType.ALGORITHM

        # Check for metric patterns
        if any(m in name_lower for m in ['准确率', 'accuracy', 'precision', 'recall', 'f1']):
            return ConceptType.METRIC

        # Check for dataset patterns
        if any(d in name_lower for d in ['数据集', 'dataset', 'mnist', 'cifar', 'imagenet']):
            return ConceptType.DATASET

        return ConceptType.UNKNOWN

    def _ai_classify(self, concept, detail_content: str = "") -> ConceptType:
        """Use AI to classify concept type."""
        prompt = f"""判断以下概念的类型：

概念名：{concept.name}
介绍：{concept.introduction}
细节内容：{detail_content[:500]}

类型选项：
- theorem: 数学定理
- definition: 形式化定义
- algorithm: 算法/流程
- model: AI/ML模型
- command: 命令/工具
- metric: 评估指标
- framework: 框架/库
- dataset: 数据集
- technique: 通用技术

只输出类型名称（例如：model）：
"""

        try:
            response = self.llm.generate(prompt).strip().lower()

            # Parse response
            for ct in ConceptType:
                if ct.value == response or ct.value in response:
                    return ct
        except Exception:
            pass

        return ConceptType.UNKNOWN

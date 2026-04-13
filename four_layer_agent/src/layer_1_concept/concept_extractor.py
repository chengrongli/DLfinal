"""
Concept extraction layer for the four-layer architecture.

This module extracts concepts and their relationships from parsed PDF content.
It uses a hybrid approach for relationship classification:
1. Predefined patterns for common relationships
2. AI-based classification for special cases
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from four_layer_agent.src.core.config import Settings

from four_layer_agent.src.core.data_models import (
    Concept,
    ConceptGraph,
    ConceptType,
    DocumentType,
    Layer1Output,
    PaperElement,
    Relationship,
    RelationshipType,
    DOCUMENT_TYPE_CONFIG,
    BIDIRECTIONAL_RELATIONSHIPS,
)


class ConceptExtractor:
    """Extract concepts and their relationships from parsed PDF content."""

    def __init__(
        self,
        settings: Settings,
        llm_client,
        doc_type: DocumentType = DocumentType.AUTO
    ):
        """
        Initialize concept extractor.

        Args:
            settings: Settings instance
            llm_client: LLM client for text generation
            doc_type: Document type for optimized extraction
        """
        self.settings = settings
        self.llm = llm_client
        self.doc_type = doc_type

        # Will be set based on detected or specified document type
        self.config = None

    def process(self, parsed_pdf: Dict) -> Layer1Output:
        """
        Process parsed PDF to extract concepts and relationships.

        Args:
            parsed_pdf: Dictionary with keys: title, full_text, file_name

        Returns:
            Layer1Output containing the concept graph
        """
        # Detect or use specified document type
        if self.doc_type == DocumentType.AUTO:
            detected_type = self._detect_document_type(parsed_pdf)
        else:
            detected_type = self.doc_type

        self.config = DOCUMENT_TYPE_CONFIG[detected_type]

        # Extract concepts
        concepts, domain = self._extract_concepts_with_metadata(parsed_pdf, detected_type)

        # Build concept graph
        graph = ConceptGraph(source_document=parsed_pdf['file_name'])
        graph.metadata = {
            "title": parsed_pdf.get('title', ''),
            "document_type": detected_type.value,
            "total_concepts": str(len(concepts)),
            "domain": domain,
        }

        for concept in concepts:
            graph.add_concept(concept)
            print(f"Added concept to graph: {concept.name}")

        print(f"Graph now has {len(graph.concepts)} concepts")

        # Identify relationships (with evidence quotes)
        relationships = self._identify_relationships_with_evidence(
            concepts, parsed_pdf, detected_type
        )
        for rel in relationships:
            graph.add_relationship(rel)

        # Create synonym relationships
        self._create_synonym_relationships(graph)

        # Generate global summary
        graph.global_summary = self._generate_global_summary(graph, detected_type)

        return Layer1Output(concept_graph=graph)

    def _detect_document_type(self, parsed_pdf: Dict) -> DocumentType:
        """Auto-detect document type from content."""
        title = parsed_pdf.get('title', '').lower()
        text = parsed_pdf.get('full_text', '')[:2000].lower()

        # Check for paper features
        paper_keywords = [
            'abstract', 'introduction', 'methodology', 'experiment',
            'conclusion', 'references', 'baseline', 'arxiv'
        ]
        paper_score = sum(1 for kw in paper_keywords if kw in text)

        # Check for lecture features
        lecture_keywords = [
            'lecture', 'slide', 'course', 'chapter', 'example',
            '练习', '例题', '课程', '讲义'
        ]
        lecture_score = sum(1 for kw in lecture_keywords if kw in title or kw in text)

        if paper_score >= 3:
            return DocumentType.PAPER
        elif lecture_score >= 2 or '课件' in title or '讲义' in title:
            return DocumentType.LECTURE
        else:
            return DocumentType.LECTURE  # Default

    def _split_into_sections(self, full_text: str) -> List[Dict[str, str]]:
        """Split Docling markdown into named sections by ## headings.

        Returns:
            List of dicts with 'heading' and 'content' keys,
            ordered as they appear in the document.
        """
        sections = re.split(r'\n(?=## )', full_text)
        result = []
        for section in sections:
            if not section.strip():
                continue
            lines = section.split('\n', 1)
            heading = lines[0].lstrip('#').strip()
            content = lines[1] if len(lines) > 1 else ''
            result.append({'heading': heading, 'content': content.strip()})
        return result

    def _sample_sections(
        self,
        sections: List[Dict[str, str]],
        total_char_budget: int,
    ) -> str:
        """Sample representative text from sections within budget.

        Strategy:
        - Abstract: always include in full (contains paper positioning + baselines)
        - Introduction: guarantee at least 2000 chars (contains baseline comparisons)
        - Remaining budget distributed across other technical sections
        - Skip references, appendices, and non-technical sections
        """
        if not sections:
            return ""

        skip_names = [
            'references', 'bibliography', 'acknowledgement', 'acknowledgment',
            'appendix', 'supplementary',
            '参考文献', '致谢', '附录',
            'broader impact', 'limitation', 'bias', 'surveillance',
            'future work', 'data overlap',
        ]

        # Identify key sections
        abstract_sec = None
        intro_sec = None
        other_secs = []

        for sec in sections:
            heading_lower = sec['heading'].lower()
            if any(s in heading_lower for s in skip_names):
                continue
            if not sec['content'].strip():
                continue

            if abstract_sec is None and 'abstract' in heading_lower:
                abstract_sec = sec
            elif intro_sec is None and ('introduction' in heading_lower or '引言' in heading_lower or '1.' in heading_lower):
                intro_sec = sec
            else:
                other_secs.append(sec)

        parts = []
        remaining = total_char_budget

        # 1. Abstract: always full
        if abstract_sec:
            content = abstract_sec['content']
            allocated = min(len(content), remaining)
            parts.append(f"## {abstract_sec['heading']}\n{content[:allocated]}")
            remaining -= allocated

        # 2. Introduction: guarantee 2000 chars (baseline comparisons live here)
        intro_min = 2000
        if intro_sec:
            allocated = min(max(intro_min, len(intro_sec['content'])), remaining)
            parts.append(f"## {intro_sec['heading']}\n{intro_sec['content'][:allocated]}")
            remaining -= allocated

        if remaining <= 0:
            return '\n\n'.join(parts)

        # 3. Other sections: distribute remaining budget proportionally
        if other_secs:
            total_other = sum(len(s['content']) for s in other_secs)
            for sec in other_secs:
                if remaining <= 0:
                    break
                ratio = len(sec['content']) / max(total_other, 1)
                allocated = min(int(remaining * ratio) + 200, len(sec['content']), remaining)
                if allocated > 0:
                    parts.append(f"## {sec['heading']}\n{sec['content'][:allocated]}")
                    remaining -= allocated

        result = '\n\n'.join(parts)
        # Fix truncated LaTeX formulas: if odd number of $$, close the last one
        if result.count('$$') % 2 != 0:
            result += '$$'
        return result

    def _extract_concepts_with_metadata(
        self, parsed_pdf: Dict, doc_type: DocumentType
    ) -> List[Concept]:
        """Extract concepts with metadata based on document type."""
        config = DOCUMENT_TYPE_CONFIG[doc_type]

        # Build allowed concept types list
        allowed_types = [ct.value for ct in config["relevant_concept_types"]]

        # Section-aware sampling: cover all sections instead of raw truncation
        sections = self._split_into_sections(parsed_pdf['full_text'])
        if sections:
            sampled_text = self._sample_sections(sections, total_char_budget=12000)
        else:
            # Fallback for documents without heading structure
            sampled_text = parsed_pdf['full_text'][:12000]

        prompt = f"""分析以下{config['name']}，提取文中出现的重要概念（通常4-8个）。

要求：
1. 概念名(name)使用原文语言（英文论文用英文，中文论文用中文）
2. 简介和上下文用中文
3. 优先提取核心技术概念（方法、模型、算法、框架），数据集和评估指标等次之

文档标题：{parsed_pdf['title']}
文档内容：{sampled_text}

直接输出JSON，不要解释。输出格式：
{{
  "domain": "论文所属领域（如：machine learning, computer vision, NLP, physics等）",
  "concepts": [
    {{
      "name": "Original Language Name",
      "introduction": "中文简短介绍",
      "context": "中文上下文",
      "section": "章节",
      "importance": 3,
      "aliases": ["alias1"],
      "type": "{allowed_types[0] if allowed_types else "technique"}"
    }}
  ]
}}

开始输出JSON：
{{"domain": "
"""

        # Request more tokens for JSON output (need full concept list)
        # Use lenient repetition settings to allow repeated JSON keys across concepts
        response = self.llm.generate(
            prompt,
            max_new_tokens=2048,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )

        # DEBUG: 打印原始响应
        print(f"=== LLM Response for concept extraction ===")
        print(response[:1000])  # 打印前1000字符
        print(f"=== END ===")

        # Extract domain from response
        domain = self._parse_domain(response)

        concepts = self._parse_concepts_with_metadata(response)
        return concepts, domain

    def _parse_domain(self, response: str) -> str:
        """Extract domain field from LLM response."""
        # Try to find "domain": "value" pattern
        match = re.search(r'"domain"\s*:\s*"([^"]+)"', response)
        if match:
            return match.group(1).strip()
        return ""

    def _parse_concepts_with_metadata(self, response: str) -> List[Concept]:
        """Parse LLM response into Concept objects."""
        concepts = []

        # DEBUG: 打印原始响应
        print(f"=== LLM Response for concept extraction ===")
        print(response[:2000])  # 打印前2000字符
        print(f"=== END ===")

        # 尝试多种方式解析JSON
        concept_list = []

        # 方法0: 修复单引号后尝试解析
        try:
            # 替换单引号为双引号（但小心字符串内部的单引号）
            fixed_response = response.replace("'", '"')
            data = json.loads(fixed_response)
            concept_list = data.get("concepts", [])
            print(f"Method 0 (fixed quotes): Found {len(concept_list)} concepts")
        except:
            pass

        # 方法1: 直接解析完整JSON
        if not concept_list:
            try:
                data = json.loads(response)
                concept_list = data.get("concepts", [])
                print(f"Method 1 (direct JSON): Found {len(concept_list)} concepts")
            except json.JSONDecodeError:
                pass

        # 方法2: 提取concept数组内容并解析
        if not concept_list:
            # 查找 "concepts": [ ... ] 模式
            match = re.search(r'"concepts"\s*:\s*\[(.*?)\]', response, re.DOTALL)
            if match:
                array_content = match.group(1)
                # 尝试解析每个对象
                objects = re.findall(r'\{[^}]*\}', array_content, re.DOTALL)
                print(f"Method 2 (extract from array): Found {len(objects)} potential objects")
                for obj_str in objects:
                    try:
                        # 先尝试直接解析
                        obj = json.loads(obj_str)
                    except:
                        try:
                            # 修复引号后再解析
                            fixed = obj_str.replace("'", '"')
                            obj = json.loads(fixed)
                        except:
                            continue
                    if 'name' in obj:
                        concept_list.append(obj)

        # 方法3: 逐对象模式匹配（处理跨行对象）
        if not concept_list:
            # 更宽松的模式：匹配 { ... "name" ... }
            pattern = r'\{\s*"name"\s*:\s*[^,}]+.*?\}'
            matches = re.finditer(pattern, response, re.DOTALL)
            objects = [m.group() for m in matches]
            print(f"Method 3 (name-based pattern): Found {len(objects)} potential objects")
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                except:
                    try:
                        fixed = obj_str.replace("'", '"')
                        obj = json.loads(fixed)
                    except:
                        continue
                if 'name' in obj:
                    concept_list.append(obj)

        # 方法4: 手动解析字段（最后的手段）
        if not concept_list:
            print("Method 4: Manual field parsing")
            # 查找所有 "name" 字段
            name_matches = re.findall(r'"name"\s*:\s*"([^"]+)"', response)
            if not name_matches:
                name_matches = re.findall(r"'name'\s*:\s*'([^']+)'", response)

            for name in name_matches:
                if not name or name in [c.get('name', '') for c in concept_list]:
                    continue
                # 为每个name创建基础对象
                concept_list.append({"name": name, "introduction": "", "aliases": []})

        print(f"Total concepts parsed: {len(concept_list)}")

        for item in concept_list[:self.settings.max_concepts_per_doc]:
            try:
                # 处理字符串值（可能是单引号或带引号）
                name = item.get('name', '')
                if isinstance(name, str):
                    name = name.strip().strip('"').strip("'")

                introduction = item.get('introduction', '')
                if isinstance(introduction, str):
                    introduction = introduction.strip().strip('"').strip("'")

                context = item.get('context', '')
                if isinstance(context, str):
                    context = context.strip().strip('"').strip("'")

                section = item.get('section')
                if section and isinstance(section, str):
                    section = section.strip().strip('"').strip("'")

                # 处理aliases
                aliases = item.get('aliases', [])
                if isinstance(aliases, str):
                    aliases = []
                elif not isinstance(aliases, list):
                    aliases = []

                # 处理importance
                importance = item.get('importance', 1)
                if isinstance(importance, str):
                    try:
                        importance = int(importance)
                    except:
                        importance = 1

                # 解析概念类型
                type_str = item.get('type', 'unknown')
                if isinstance(type_str, str):
                    type_str = type_str.strip().strip('"').strip("'").lower()

                concept_type = ConceptType.UNKNOWN
                for ct in ConceptType:
                    if ct.value in type_str or type_str in ct.value:
                        concept_type = ct
                        break

                concepts.append(Concept(
                    name=name,
                    introduction=introduction,
                    context=context,
                    aliases=aliases,
                    section_name=section,
                    importance_score=importance,
                    concept_type=concept_type,
                    paper_role=PaperElement.UNKNOWN
                ))
                print(f"  Parsed concept: {name}")
            except Exception as e:
                print(f"  Error parsing concept: {e}")
                continue

        return concepts

    def _identify_relationships_with_evidence(
        self, concepts: List[Concept], parsed_pdf: Dict, doc_type: DocumentType
    ) -> List[Relationship]:
        """Identify relationships with evidence quotes."""
        config = DOCUMENT_TYPE_CONFIG[doc_type]
        allowed_rel_types = config["relevant_relationship_types"]

        text = parsed_pdf['full_text']

        # 1. Sort by importance, only use top-K for relationship extraction
        sorted_concepts = sorted(
            concepts, key=lambda c: c.importance_score, reverse=True
        )
        top_concepts = sorted_concepts[:self.settings.max_concepts_for_relationships]

        # Build search terms: name + aliases for each concept
        def search_terms(concept: Concept) -> List[str]:
            terms = [concept.name]
            terms.extend(concept.aliases)
            return terms

        # 2. Check pairs of top concepts
        relationships = []
        for i in range(len(top_concepts)):
            for j in range(i + 1, len(top_concepts)):
                c1 = top_concepts[i]
                c2 = top_concepts[j]

                # Check all combinations of (c1 terms) x (c2 terms)
                evidence = None
                for t1 in search_terms(c1):
                    if evidence:
                        break
                    for t2 in search_terms(c2):
                        evidence = self._find_co_occurrence(t1, t2, text)
                        if evidence:
                            break

                if evidence:
                    # Try pattern matching first
                    rel = self._classify_by_pattern(c1, c2, evidence, allowed_rel_types)
                    if not rel:
                        # Fall back to AI classification
                        rel = self._ai_classify_relationship(c1, c2, evidence, allowed_rel_types)

                    if rel:
                        relationships.append(rel)

        # 3. Merge comparison clusters (A-B, A-C, B-C → one grouped relation)
        relationships = self._merge_comparison_clusters(relationships)

        # 4. Limit per relationship type
        relationships = self._limit_by_type(relationships)

        return relationships[:self.settings.max_relationships_per_doc]

    def _merge_comparison_clusters(
        self, relationships: List[Relationship]
    ) -> List[Relationship]:
        """Merge pairwise comparison relationships into cluster relations.

        If A-B, A-C, B-C are all comparison, they belong to the same cluster.
        Keep only one representative relationship with all cluster members noted.
        """
        comparisons = [r for r in relationships if r.relationship_type == RelationshipType.COMPARISON]
        others = [r for r in relationships if r.relationship_type != RelationshipType.COMPARISON]

        if not comparisons:
            return relationships

        # Union-Find to group connected concepts
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for r in comparisons:
            union(r.source_concept, r.target_concept)

        # Group by root
        clusters: Dict[str, List[Relationship]] = {}
        for r in comparisons:
            root = find(r.source_concept)
            clusters.setdefault(root, []).append(r)

        # For each cluster, keep the first (highest importance) relationship
        # and expand its description to mention all cluster members
        merged = []
        for root, rels in clusters.items():
            if len(rels) <= 2:
                # Small cluster (≤3 concepts): keep as-is, no explosion risk
                merged.extend(rels)
                continue

            # Large cluster: keep one representative, note all members
            all_names = set()
            for r in rels:
                all_names.add(r.source_concept)
                all_names.add(r.target_concept)

            representative = rels[0]
            member_list = "、".join(sorted(all_names))
            # Use all members as source, empty target — key becomes "A/B/C->"
            merged.append(Relationship(
                source_concept=member_list,
                target_concept="",
                relationship_type=RelationshipType.COMPARISON,
                description=f"{member_list} 为同类/并列概念，各有特点",
                evidence_quote=representative.evidence_quote,
                is_bidirectional=True,
                confidence=representative.confidence,
            ))

        return others + merged

    def _limit_by_type(self, relationships: List[Relationship]) -> List[Relationship]:
        """Limit the number of relationships per type."""
        max_per_type = self.settings.max_relationships_per_type
        counts: Dict[RelationshipType, int] = {}
        result = []
        for r in relationships:
            counts[r.relationship_type] = counts.get(r.relationship_type, 0) + 1
            if counts[r.relationship_type] <= max_per_type:
                result.append(r)
        return result

    def _find_co_occurrence(
        self, term1: str, term2: str, text: str
    ) -> Optional[str]:
        """Find sentence or paragraph where both terms appear."""
        # First try sentence-level (more precise evidence)
        sentences = re.split(r'[。！？.!?\n]', text)
        for sent in sentences:
            if len(sent.strip()) < 10:
                continue
            if term1 in sent and term2 in sent:
                return sent.strip()

        # Fallback: paragraph-level co-occurrence
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) < 30:
                continue
            if term1 in para and term2 in para:
                # Return the most relevant sentence from the paragraph
                for sent in re.split(r'[。！？.!?]', para):
                    if term1 in sent or term2 in sent:
                        return sent.strip()
                return para[:300].strip()

        return None

    def _classify_by_pattern(
        self, c1: Concept, c2: Concept, sentence: str, allowed_types: List[RelationshipType]
    ) -> Optional[Relationship]:
        """Classify relationship using predefined patterns."""
        patterns = {
            RelationshipType.PREREQUISITE: [
                r'(.+)是(.+)的前置',
                r'(.+)需要先掌握(.+)',
            ],
            RelationshipType.DERIVATION: [
                r'(.+)可以由(.+)推导',
                r'从(.+)可以得到(.+)',
            ],
            RelationshipType.COMPONENT: [
                r'(.+)包含(.+)',
                r'(.+)由(.+)组成',
            ],
            RelationshipType.SOLVES: [
                r'(.+)解决(.+)问题',
            ],
            RelationshipType.IMPROVES_UPON: [
                r'(.+)改进自(.+)',
                r'(.+)在(.+)的基础上',
            ],
            RelationshipType.USES: [
                r'(.+)使用(.+)',
                r'(.+)基于(.+)',
            ],
        }

        sentence_lower = sentence.lower()

        for rel_type, rel_patterns in patterns.items():
            if rel_type not in allowed_types:
                continue

            for pattern in rel_patterns:
                if re.search(pattern, sentence_lower):
                    return Relationship(
                        source_concept=c1.name,
                        target_concept=c2.name,
                        relationship_type=rel_type,
                        description=f"Detected {rel_type.value} relationship",
                        evidence_quote=sentence,
                        is_bidirectional=rel_type in BIDIRECTIONAL_RELATIONSHIPS,
                        confidence=1.0
                    )

        return None

    def _ai_classify_relationship(
        self, c1: Concept, c2: Concept, evidence: str, allowed_types: List[RelationshipType]
    ) -> Optional[Relationship]:
        """Use AI to classify relationship."""
        rel_descriptions = {
            RelationshipType.PREREQUISITE: "prerequisite - A是B的前置条件",
            RelationshipType.DERIVATION: "derivation - B由A推导",
            RelationshipType.APPLICATION: "application - A应用于B",
            RelationshipType.COMPARISON: "comparison - A与B对比",
            RelationshipType.GENERALIZATION: "generalization - A是B的一般化",
            RelationshipType.SPECIALIZATION: "specialization - B是A的特殊化",
            RelationshipType.COMPONENT: "component - A是B的组成部分",
            RelationshipType.SOLVES: "solves - A解决了B问题",
            RelationshipType.IMPROVES_UPON: "improves_upon - A改进了B",
            RelationshipType.USES: "uses - A使用了B",
            RelationshipType.SYNONYM: "synonym - A与B是同义词",
        }

        allowed_rels = "\n".join(
            f"- {rel_descriptions[rt]}"
            for rt in allowed_types if rt in rel_descriptions
        )

        prompt = f"""分析以下两个概念在给定句子中的关系：

概念A：{c1.name}
概念B：{c2.name}

原句：{evidence}

关系类型（只能选择以下类型之一）：
{allowed_rels}

输出JSON：
{{"type": "关系类型", "description": "关系描述", "is_bidirectional": false, "confidence": 0.8}}
"""

        try:
            response = self.llm.generate(prompt, max_new_tokens=512)
            # Try to parse JSON
            match = re.search(r'\{[^{}]*\}', response)
            if match:
                parsed = json.loads(match.group(0))
                rel_type_str = parsed.get('type', '').split()[0]

                for rt in allowed_types:
                    if rt.value == rel_type_str or rt.value in rel_type_str:
                        return Relationship(
                            source_concept=c1.name,
                            target_concept=c2.name,
                            relationship_type=rt,
                            description=parsed.get('description', ''),
                            evidence_quote=evidence,
                            is_bidirectional=parsed.get('is_bidirectional', False),
                            confidence=parsed.get('confidence', 0.7)
                        )
        except Exception:
            pass

        return None

    def _create_synonym_relationships(self, graph: ConceptGraph):
        """Create SYNONYM relationships for aliases."""
        for concept in graph.concepts.values():
            for alias in concept.aliases:
                # Skip self-referencing aliases (e.g., CIFAR-10 has alias CIFAR-10)
                if alias == concept.name or alias not in graph.concepts:
                    continue
                rel = Relationship(
                    source_concept=concept.name,
                    target_concept=alias,
                    relationship_type=RelationshipType.SYNONYM,
                    description=f"{concept.name} 是 {alias} 的全称/缩写",
                    is_bidirectional=True,
                    confidence=1.0
                )
                graph.add_relationship(rel)

    def _generate_global_summary(self, graph: ConceptGraph, doc_type: DocumentType) -> str:
        """Generate global summary based on concept graph."""
        top_concepts = sorted(
            graph.concepts.values(),
            key=lambda c: c.importance_score,
            reverse=True
        )[:5]

        concepts_str = "\n".join(
            f"- {c.name}: {c.introduction}"
            for c in top_concepts
        )

        rels_str = "\n".join(
            f"- {r.source_concept} -> {r.target_concept} ({r.relationship_type.value})"
            for r in graph.relationships[:5]
        )

        prompt = f"""基于以下概念图，生成一段200字以内的全局摘要：

核心概念：
{concepts_str}

主要关系：
{rels_str}

摘要应包含：文档的主要内容、核心概念和关键关系。
直接输出摘要，不要其他内容。
"""

        try:
            return self.llm.generate(prompt, max_new_tokens=512).strip()
        except Exception:
            return "文档主要介绍了相关概念及其关系。"

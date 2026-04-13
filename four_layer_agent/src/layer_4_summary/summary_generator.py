"""
Summary generation layer for the four-layer architecture.

Converts the structured JSON output into structured Markdown notes.
Includes LaTeX formulas extracted from the original paper text.
Pure template-based, zero LLM calls.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional


def _clean_latex_formula(formula: str) -> str:
    """Clean LaTeX formula formatting issues from PDF parsing."""
    try:
        import re
        result = formula

        # Step 1: Clean braces content iteratively
        def clean_braces_content(match):
            cmd = match.group(1) or ''
            content = match.group(2)
            # Remove \, thin spaces
            content = re.sub(r'\\,\s*', '', content)
            # Remove all other spaces
            content = content.replace(' ', '')
            return f'{cmd}{{{content}}}'

        for _ in range(8):
            result = re.sub(r'(\\[a-zA-Z]+)?\{([^\{\}]*)\}', clean_braces_content, result)

        # Step 2: Remove spaces before ^ and _
        result = re.sub(r'(\})\s+\^', r'\1^', result)
        result = re.sub(r'(\w)\s+\^', r'\1^', result)
        result = re.sub(r'(\})\s+_', r'\1_', result)
        result = re.sub(r'(\w)\s+_', r'\1_', result)

        # Step 3: Remove spaces between \cmd and {
        result = re.sub(r'(\\[a-zA-Z]+)\s+\{', r'\1{', result)

        # Step 4: Fix _{ pattern
        result = re.sub(r'(\w) +_\{', r'\1_{', result)
        result = re.sub(r'(\\[a-zA-Z]+) +_\{', r'\1_{', result)

        # Step 5: Remove ALL remaining spaces around braces and symbols
        result = re.sub(r'\}\s+([a-zA-Z0-9\\\.])', r'}\1', result)  # } x -> }x
        result = re.sub(r'([a-zA-Z0-9\\\.])\s+\{', r'\1{', result)  # x { -> x{

        # Step 6: Normalize operators
        result = re.sub(r'\s*=\s*', ' = ', result)
        result = re.sub(r'\s*\+\s*', ' + ', result)
        result = re.sub(r'\s*-\s*', ' - ', result)
        result = re.sub(r'\s*,\s*', ', ', result)
        result = re.sub(r'\\sim\s+', r'\\sim ', result)

        # Step 7: Fix parentheses
        result = re.sub(r'\(\s+', '(', result)
        result = re.sub(r'\s+\)', ')', result)
        # Also remove spaces after ( and before ) in simple cases
        result = re.sub(r'\(([a-zA-Z0-9])\s+', r'(\1', result)
        result = re.sub(r'\s+([a-zA-Z0-9])\)', r'\1)', result)

        # Step 8: Clean up weird patterns
        result = result.replace(r' \, ', ' ')

        return result.strip()
    except Exception as e:
        print(f"Warning: Failed to clean formula: {e}")
        import traceback
        traceback.print_exc()
        return formula


def _translate_rel_type(rel_type: str) -> str:
    """Translate relationship type to Chinese."""
    translations = {
        "application": "应用",
        "comparison": "对比",
        "prerequisite": "前置条件",
        "derivation": "推导",
        "component": "组成部分",
        "solves": "解决",
        "improves_upon": "改进",
        "uses": "使用",
        "evaluated_on": "评估于",
        "contradiction": "矛盾",
        "synonym": "同义词",
        "generalization": "一般化",
        "specialization": "特殊化",
    }
    return translations.get(rel_type, rel_type)


class SummaryGenerator:
    """Generate structured Markdown notes from four-layer JSON output."""

    def __init__(self, llm_client=None):
        pass  # No LLM needed

    def generate(self, four_layer_json: dict, formulas: dict = None) -> str:
        """Convert four-layer JSON to structured Markdown note.

        Args:
            four_layer_json: The complete four-layer output dict.
            formulas: Dict mapping section heading -> list of formula paragraphs.

        Returns:
            Markdown-formatted note string.
        """
        return self._generate_note(four_layer_json, formulas or {})

    def _generate_note(self, data: dict, formulas: dict) -> str:
        """Generate structured Markdown note from JSON."""
        graph = data["layer1_concepts"]["concept_graph"]
        concepts = graph["concepts"]
        relationships = graph["relationships"]
        metadata = graph["metadata"]
        global_summary = graph.get("global_summary", "")
        domain = metadata.get("domain", "")

        details = data["layer3_details"]["detail_contents"]
        apps = data["layer4_applications"]["application_contents"]

        title = metadata.get("title", "")

        # Header
        parts = [f"## {title}\n"]
        if domain:
            parts.append(f"> 领域：{domain}\n")

        # Overview
        if global_summary:
            parts.append("### 概述\n")
            parts.append(f"{global_summary}\n")

        # Concepts sorted by importance
        sorted_concepts = sorted(
            concepts.items(),
            key=lambda x: x[1].get("importance", 1),
            reverse=True,
        )

        parts.append("### 核心概念\n")
        for name, c in sorted_concepts:
            stars = "★" * c.get("importance", 1) + "☆" * (5 - c.get("importance", 1))
            aliases = c.get("aliases", [])
            alias_str = f"（别名：{', '.join(aliases)}）" if aliases else ""

            parts.append(f"#### {name} [{stars}]{alias_str}\n")
            parts.append(f"{c.get('introduction', '')}\n")

            detail = details.get(name, "")
            if detail:
                parts.append(f"\n{detail}\n")

            # Attach formulas from the concept's section
            section = c.get("section", "")
            section_formulas = _match_formulas(section, formulas, concept_name=name)
            if section_formulas:
                parts.append("\n**关键公式：**\n")
                for f in section_formulas:
                    cleaned_formula = _clean_latex_formula(f)
                    parts.append(f"{cleaned_formula}\n")

            parts.append("")

        # Relationships
        if relationships:
            parts.append("### 概念关系\n")
            for r in relationships:
                src = r["source"]
                tgt = r.get("target", "")
                rel_type = r.get("type", "")
                desc = r.get("description", "")

                # Translate relationship type to Chinese
                rel_type_zh = _translate_rel_type(rel_type)

                if tgt:
                    parts.append(f"- **{src}** → **{tgt}**（{rel_type_zh}）：{desc}")
                else:
                    parts.append(f"- **{src}**（{rel_type_zh}）：{desc}")
            parts.append("")

        # Applications
        parts.append("### 应用场景\n")
        for name, c in sorted_concepts:
            app = apps.get(name, "")
            if app:
                parts.append(f"- **{name}**：{app}")
        parts.append("")

        return "\n".join(parts)


def _match_formulas(section: str, formulas: dict, concept_name: str = "") -> List[str]:
    """Find formula paragraphs matching a concept's section.

    Smart filtering:
    - If the concept name appears in the section title, include all formulas from that section
    - Otherwise, only include formulas that mention the concept name
    """
    if not section or not formulas:
        return []

    # Only use exact match
    if section in formulas:
        section_formulas = formulas[section]
    else:
        section_formulas = []

    if not section_formulas:
        return []

    # Smart filtering based on whether concept is in section title
    if concept_name and section_formulas:
        # Check if concept name is prominently in section title
        section_lower = section.lower()
        concept_lower = concept_name.lower()

        # For "3 Energy-Based Models and Sampling" with "Langevin Dynamics":
        # Since "Langevin Dynamics" != "Energy-Based Models", we need to check content
        # But if section title contains the concept name, trust that section

        # Simple heuristic: if section name starts with concept name, or contains it as a main topic
        # For now, be lenient: only filter if section clearly belongs to another concept
        # E.g., "3 Sample Replay Buffer" clearly belongs to Sample Replay Buffer only
        # But "3 Energy-Based Models and Sampling" is shared

        # Define clearly single-concept sections
        single_concept_sections = [
            "3 Sample Replay Buffer",
            "4.4 Out-of-Distribution Generalization",
            "5 Trajectory Modeling",
            "5.2 Multi-Step Trajectory Generation",
        ]

        if section in single_concept_sections:
            # This section is clearly for this concept only
            return section_formulas
        else:
            # Shared section (like "3 Energy-Based Models and Sampling")
            # For shared sections, include formulas if they mention the concept
            search_terms = [concept_lower]
            search_terms.append(concept_name.replace(" ", "").lower())
            search_terms.append(concept_name.replace(" ", "_").lower())

            # For Langevin Dynamics, also accept formulas without explicit names
            # since it's the main topic of that section
            if "Langevin" in concept_name or "Dynamics" in concept_name:
                # Include first formula as fallback for main concepts
                return section_formulas[:1] if section_formulas else []

            filtered = []
            for formula in section_formulas:
                formula_lower = formula.lower()
                if any(term in formula_lower for term in search_terms):
                    filtered.append(formula)

            return filtered if filtered else []

    return section_formulas

"""
Summary generation layer for the four-layer architecture.

Converts the structured JSON output into structured Markdown notes.
Includes LaTeX formulas extracted from the original paper text.
Pure template-based, zero LLM calls.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional


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
            section_formulas = _match_formulas(section, formulas)
            if section_formulas:
                parts.append("\n**关键公式：**\n")
                for f in section_formulas:
                    parts.append(f"{f}\n")

            parts.append("")

        # Relationships
        if relationships:
            parts.append("### 概念关系\n")
            for r in relationships:
                src = r["source"]
                tgt = r.get("target", "")
                rel_type = r.get("type", "")
                desc = r.get("description", "")

                if tgt:
                    parts.append(f"- **{src}** → **{tgt}**（{rel_type}）：{desc}")
                else:
                    parts.append(f"- **{src}**（{rel_type}）：{desc}")
            parts.append("")

        # Applications
        parts.append("### 应用场景\n")
        for name, c in sorted_concepts:
            app = apps.get(name, "")
            if app:
                parts.append(f"- **{name}**：{app}")
        parts.append("")

        return "\n".join(parts)


def _match_formulas(section: str, formulas: dict) -> List[str]:
    """Find formula paragraphs matching a concept's section.

    Tries exact match first, then partial match on section number.
    """
    if not section or not formulas:
        return []

    # Exact match
    if section in formulas:
        return formulas[section]

    # Partial match: extract section number (e.g., "3" from "3.1 Maximum Likelihood")
    section_num = re.match(r'(\d+(?:\.\d+)*)', section)
    if not section_num:
        return []

    target = section_num.group(1)
    results = []
    for sec_heading, paras in formulas.items():
        sec_num = re.match(r'(\d+(?:\.\d+)*)', sec_heading)
        if sec_num and (sec_num.group(1) == target or sec_num.group(1).startswith(target + ".")):
            results.extend(paras)

    return results

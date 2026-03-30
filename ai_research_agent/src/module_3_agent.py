from __future__ import annotations

import requests

from ai_research_agent.src.config import Settings
from ai_research_agent.src.utils import append_jsonl, get_logger


class ThreeLayerSummaryBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)

    def _call_llm(self, prompt: str) -> str:
        if not self.settings.llm_api_base or not self.settings.llm_api_key:
            return ""

        endpoint = self.settings.llm_api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.llm_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            self.logger.warning("LLM call failed, fallback to template summary: %s", exc)
            return ""

    def _fallback_summary(self, text: str, title: str) -> dict:
        short = text[:1600]
        return {
            "concept_layer": (
                f"文献《{title}》主要围绕相关主题展开，核心内容可概括为：\n"
                f"{short[:400]}..."
            ),
            "detail_layer": (
                "从定义到证明通常遵循：先提出问题与符号体系，再给出命题与假设，"
                "最后通过逐步推导得到结论。建议在阅读时重点标记每个结论对应的前提条件。"
            ),
            "application_layer": (
                "潜在应用方向包括：构建教学材料、任务自动化系统、以及在相似问题上迁移验证。"
            ),
        }

    def build_three_layers(
        self,
        title: str,
        full_text: str,
        search_context: dict | None = None,
    ) -> dict:
        search_block = ""
        if search_context:
            lines = []
            for term, rows in search_context.items():
                lines.append(f"Term: {term}")
                for row in rows:
                    lines.append(f"- {row.get('title', '')}: {row.get('snippet', '')}")
            search_block = "\n".join(lines)

        prompt = f"""
你是研究助理，请基于以下论文内容输出三层总结，使用中文：
1. 概念层：这篇文献主要讲了什么。
2. 细节层：核心定义、命题、证明或推导逻辑如何展开。
3. 应用层：可能的应用方向与后续研究问题。

输出格式必须是 JSON，键为 concept_layer, detail_layer, application_layer。

论文标题：{title}
论文内容：
{full_text[:12000]}

可选联网补充：
{search_block[:4000]}
""".strip()

        model_response = self._call_llm(prompt)
        if not model_response:
            return self._fallback_summary(full_text, title)

        try:
            # Prefer direct JSON content from model.
            import json

            return json.loads(model_response)
        except Exception:
            # Keep raw output while preserving schema shape.
            return {
                "concept_layer": model_response,
                "detail_layer": "模型输出非JSON，已将原始输出存入概念层。",
                "application_layer": "请调整提示词或模型输出格式约束后重试。",
            }

    def to_sft_row(self, title: str, summary: dict) -> dict:
        instruction = (
            "请对给定文献输出三层总结，包含概念层、细节层、应用层，要求结构清晰且可用于教学。"
        )
        output = (
            f"标题：{title}\n"
            f"概念层：{summary.get('concept_layer', '')}\n"
            f"细节层：{summary.get('detail_layer', '')}\n"
            f"应用层：{summary.get('application_layer', '')}"
        )
        return {
            "instruction": instruction,
            "input": title,
            "output": output,
        }

    def export_dataset(self, rows: list[dict]) -> None:
        dataset_path = self.settings.dataset_dir / self.settings.dataset_file_name
        append_jsonl(dataset_path, rows)
        self.logger.info("SFT dataset exported: %s", dataset_path)

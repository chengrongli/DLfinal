from __future__ import annotations

import json
import re
from typing import Any

from ai_research_agent.src.config import Settings
from ai_research_agent.src.prompts.summary_templates import (
    APPLICATION_LEVEL_PROMPT,
    CONCEPT_LEVEL_PROMPT,
    DETAIL_LEVEL_PROMPT,
)
from ai_research_agent.src.utils import append_jsonl, get_logger


class ThreeLayerSummaryBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)
        self._tokenizer: Any = None
        self._model: Any = None

    def _load_local_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except Exception:  # pragma: no cover
            BitsAndBytesConfig = None

        self.logger.info("Loading local model: %s", self.settings.llm_model_name)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.settings.llm_model_name,
                trust_remote_code=True,
                use_fast=False,
            )

            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
            }

            if self.settings.llm_local_use_4bit and BitsAndBytesConfig is not None and torch.cuda.is_available():
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            self._model = AutoModelForCausalLM.from_pretrained(
                self.settings.llm_model_name,
                **model_kwargs,
            )
        except Exception as exc:
            self.logger.warning(
                "Transformers local load failed (%s). Falling back to Unsloth loader.",
                exc,
            )
            from unsloth import FastLanguageModel

            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.settings.llm_model_name,
                max_seq_length=self.settings.llm_local_prompt_max_tokens,
                dtype=None,
                load_in_4bit=self.settings.llm_local_use_4bit,
            )
            try:
                FastLanguageModel.for_inference(self._model)
            except Exception:
                pass

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _call_llm(self, prompt: str) -> str:
        try:
            import torch

            self._load_local_model()
            assert self._tokenizer is not None and self._model is not None

            tokenized = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.settings.llm_local_prompt_max_tokens,
            )
            model_device = next(self._model.parameters()).device
            tokenized = {k: v.to(model_device) for k, v in tokenized.items()}

            with torch.no_grad():
                generated = self._model.generate(
                    **tokenized,
                    max_new_tokens=self.settings.llm_local_max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            prompt_len = tokenized["input_ids"].shape[-1]
            output_ids = generated[0][prompt_len:]
            return self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            self.logger.warning("Local model generation failed, fallback to template summary: %s", exc)
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

        prompt = (
            f"{CONCEPT_LEVEL_PROMPT}\n\n"
            f"{DETAIL_LEVEL_PROMPT}\n\n"
            f"{APPLICATION_LEVEL_PROMPT}\n\n"
            "输出格式必须是 JSON，键为 concept_layer, detail_layer, application_layer。\n\n"
            f"论文标题：{title}\n"
            "论文内容：\n"
            f"{full_text[:12000]}\n\n"
            "可选联网补充：\n"
            f"{search_block[:4000]}"
        ).strip()

        model_response = self._call_llm(prompt)
        if not model_response:
            return self._fallback_summary(full_text, title)

        try:
            return json.loads(model_response)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", model_response)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass

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

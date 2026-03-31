from __future__ import annotations

import json
import re
from pathlib import Path
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
            from transformers import StoppingCriteria, StoppingCriteriaList

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

            stop_ids = self._tokenizer.encode("\n", add_special_tokens=False)

            class _StopOnTokens(StoppingCriteria):
                def __init__(self, tokens: list[int]) -> None:
                    self.tokens = set(tokens)

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    if not self.tokens:
                        return False
                    return int(input_ids[0, -1]) in self.tokens

            stopping_criteria = None
            if stop_ids:
                stopping_criteria = StoppingCriteriaList([_StopOnTokens(stop_ids)])

            with torch.no_grad():
                generated = self._model.generate(
                    **tokenized,
                    max_new_tokens=min(self.settings.llm_local_max_new_tokens, 100),
                    do_sample=False,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=6,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            prompt_len = tokenized["input_ids"].shape[-1]
            output_ids = generated[0][prompt_len:]
            return self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            self.logger.warning("Local model generation failed, fallback to template summary: %s", exc)
            return ""

    def _clip_text(self, text: str, max_chars: int = 100, max_words: int = 100) -> str:
        text = text.strip()
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]).strip()
        if len(text) > max_chars:
            text = text[:max_chars].strip()
        return text

    def _normalize_text(self, text: str) -> str:
        lines: list[str] = []
        in_code_block = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("$"):
                continue
            if stripped:
                lines.append(stripped)
        cleaned = " ".join(lines)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _parse_response(self, text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass

        labels = {
            "concept_layer": ["概念层", "concept_layer", "concept"],
            "detail_layer": ["细节层", "detail_layer", "detail"],
            "application_layer": ["应用层", "application_layer", "application"],
        }
        label_pattern = r"(" + "|".join(
            re.escape(label)
            for group in labels.values()
            for label in group
        ) + r")\s*[:：]"

        matches = list(re.finditer(label_pattern, text, flags=re.IGNORECASE))
        if matches:
            extracted = {
                "concept_layer": "",
                "detail_layer": "",
                "application_layer": "",
            }
            for index, match in enumerate(matches):
                start = match.end()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                content = text[start:end].strip()
                label = match.group(1).lower()
                for key, variants in labels.items():
                    if any(label == variant.lower() for variant in variants):
                        if content:
                            extracted[key] = content
                        break
            return extracted

        cleaned = self._normalize_text(text)
        if not cleaned:
            return {
                "concept_layer": "",
                "detail_layer": "",
                "application_layer": "",
            }
        chunks = [cleaned[i : i + 100] for i in range(0, min(len(cleaned), 300), 100)]
        while len(chunks) < 3:
            chunks.append("")
        return {
            "concept_layer": chunks[0],
            "detail_layer": chunks[1],
            "application_layer": chunks[2],
        }

    def _extract_layer_text(self, labels: list[str], text: str) -> str:
        label_pattern = r"(" + "|".join(re.escape(label) for label in labels) + r")\s*[:：]"
        match = re.search(label_pattern, text, flags=re.IGNORECASE)
        if match:
            return text[match.end():].strip()
        cleaned = self._normalize_text(text)
        if not cleaned:
            return ""
        return cleaned

    def _first_line(self, text: str) -> str:
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned:
                return cleaned
        return ""

    def _save_raw_response(self, doc_id: str, text: str) -> None:
        if not doc_id:
            return
        output_dir = Path(self.settings.summaries_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / f"{doc_id}.raw.txt"
        raw_path.write_text(text or "", encoding="utf-8")

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
        doc_id: str,
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

        context_block = (
            f"论文标题：{title}\n"
            "论文内容：\n"
            f"{full_text[:12000]}\n\n"
            "可选联网补充：\n"
            f"{search_block[:4000]}"
        ).strip()

        concept_raw = self._call_llm(f"{CONCEPT_LEVEL_PROMPT}\n\n{context_block}")
        detail_raw = self._call_llm(f"{DETAIL_LEVEL_PROMPT}\n\n{context_block}")
        application_raw = self._call_llm(f"{APPLICATION_LEVEL_PROMPT}\n\n{context_block}")

        concept_clean = self._first_line(concept_raw)
        detail_clean = self._first_line(detail_raw)
        application_clean = self._first_line(application_raw)

        combined_raw = (
            "[concept_layer]\n"
            f"{concept_clean}\n\n"
            "[detail_layer]\n"
            f"{detail_clean}\n\n"
            "[application_layer]\n"
            f"{application_clean}"
        )
        self._save_raw_response(doc_id, combined_raw)

        if not (concept_raw or detail_raw or application_raw):
            return self._fallback_summary(full_text, title)

        summary = {
            "concept_layer": self._clip_text(
                self._extract_layer_text(["概念层", "concept_layer", "concept"], concept_clean)
            ),
            "detail_layer": self._clip_text(
                self._extract_layer_text(["细节层", "detail_layer", "detail"], detail_clean)
            ),
            "application_layer": self._clip_text(
                self._extract_layer_text(["应用层", "application_layer", "application"], application_clean)
            ),
        }
        return summary

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

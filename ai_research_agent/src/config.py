from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    raw_pdf_dir: Path = PROJECT_ROOT / "data" / "raw_pdfs"
    parsed_md_dir: Path = PROJECT_ROOT / "data" / "parsed_mds"
    dataset_dir: Path = PROJECT_ROOT / "data" / "dataset"

    parsed_json_suffix: str = ".parsed.json"
    summary_json_suffix: str = ".summary.json"
    dataset_file_name: str = "sft_data.jsonl"

    search_max_results: int = 5
    search_terms_per_doc: int = 4

    # For optional OpenAI-compatible API usage.
    llm_api_base: str = os.getenv("LLM_API_BASE", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF")


def get_settings() -> Settings:
    settings = Settings()
    settings.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.parsed_md_dir.mkdir(parents=True, exist_ok=True)
    settings.dataset_dir.mkdir(parents=True, exist_ok=True)
    return settings

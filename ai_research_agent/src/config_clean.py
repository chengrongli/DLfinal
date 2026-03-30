from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    raw_pdf_dir: Path = PROJECT_ROOT / "data" / "raw_pdfs"
    parsed_md_dir: Path = PROJECT_ROOT / "data" / "parsed_mds"
    search_cache_dir: Path = PROJECT_ROOT / "data" / "search_cache"
    summaries_dir: Path = PROJECT_ROOT / "data" / "summaries"
    dataset_dir: Path = PROJECT_ROOT / "data" / "dataset"

    parsed_json_suffix: str = ".parsed.json"
    summary_json_suffix: str = ".summary.json"
    dataset_file_name: str = "sft_data.jsonl"

    search_max_results: int = 5
    search_terms_per_doc: int = 4

    # Local model generation settings.
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-8B")
    llm_local_max_new_tokens: int = int(os.getenv("LLM_LOCAL_MAX_NEW_TOKENS", "512"))
    llm_local_prompt_max_tokens: int = int(os.getenv("LLM_LOCAL_PROMPT_MAX_TOKENS", "4096"))
    llm_local_use_4bit: bool = os.getenv("LLM_LOCAL_USE_4BIT", "1") == "1"


def get_settings() -> Settings:
    settings = Settings()
    settings.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.parsed_md_dir.mkdir(parents=True, exist_ok=True)
    settings.search_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.summaries_dir.mkdir(parents=True, exist_ok=True)
    settings.dataset_dir.mkdir(parents=True, exist_ok=True)
    return settings

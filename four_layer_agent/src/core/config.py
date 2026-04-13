"""
Configuration management for the four-layer architecture.

This module defines all configuration settings for the four-layer agent system.
Settings can be overridden via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from four_layer_agent.src.core.data_models import DocumentType


# four_layer_agent directory (parent of src/)
FOUR_LAYER_ROOT = Path(__file__).resolve().parents[2]
# Project root (parent of four_layer_agent)
PROJECT_ROOT = FOUR_LAYER_ROOT.parent


@dataclass
class Settings:
    """Configuration settings for the four-layer agent system"""

    # ===== Paths =====
    raw_pdfs_dir: Path = FOUR_LAYER_ROOT / "data" / "raw_pdfs"
    four_layer_output_dir: Path = FOUR_LAYER_ROOT / "data" / "output"

    # ===== File suffixes =====
    concept_graph_suffix: str = ".concept_graph.json"
    layer1_suffix: str = ".layer1.json"
    layer2_suffix: str = ".layer2.json"
    layer3_suffix: str = ".layer3.json"
    layer4_suffix: str = ".layer4.json"
    final_output_suffix: str = ".four_layer.json"

    # ===== Embedding settings =====
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_cache_size: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))

    # ===== Relationship classification settings =====
    enable_ai_relationships: bool = os.getenv("ENABLE_AI_RELATIONSHIPS", "1") == "1"
    relationship_confidence_threshold: float = float(os.getenv("RELATIONSHIP_CONFIDENCE_THRESHOLD", "0.7"))

    # ===== Concept extraction settings =====
    max_concepts_per_doc: int = int(os.getenv("MAX_CONCEPTS_PER_DOC", "30"))
    max_relationships_per_doc: int = int(os.getenv("MAX_RELATIONSHIPS_PER_DOC", "50"))
    max_concepts_for_relationships: int = int(os.getenv("MAX_CONCEPTS_FOR_RELATIONSHIPS", "15"))
    max_relationships_per_type: int = int(os.getenv("MAX_RELATIONSHIPS_PER_TYPE", "5"))

    # ===== Default document type =====
    default_doc_type: DocumentType = DocumentType(
        os.getenv("DEFAULT_DOC_TYPE", "auto")
    )

    # ===== LLM settings =====
    # Use absolute path, ignore env var if it's relative
    _llm_model_env = os.getenv("LLM_MODEL_NAME", "")
    if _llm_model_env and not Path(_llm_model_env).is_absolute():
        llm_model_name: str = str(PROJECT_ROOT / "qwen3")
    else:
        llm_model_name: str = _llm_model_env or str(PROJECT_ROOT / "qwen3")
    llm_local_max_new_tokens: int = int(os.getenv("LLM_LOCAL_MAX_NEW_TOKENS", "1024"))
    llm_local_prompt_max_tokens: int = int(os.getenv("LLM_LOCAL_PROMPT_MAX_TOKENS", "8192"))
    llm_local_use_4bit: bool = os.getenv("LLM_LOCAL_USE_4BIT", "1") == "1"

    # ===== Processing settings =====
    enable_caching: bool = os.getenv("ENABLE_CACHING", "1") == "1"
    overwrite_cache: bool = os.getenv("OVERWRITE_CACHE", "0") == "1"


def get_settings() -> Settings:
    """Get settings instance and create necessary directories"""
    settings = Settings()

    # Create directories
    settings.raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
    settings.four_layer_output_dir.mkdir(parents=True, exist_ok=True)

    return settings


def update_settings_from_dict(settings: Settings, updates: dict) -> Settings:
    """Update settings from a dictionary (useful for CLI overrides)"""
    for key, value in updates.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings

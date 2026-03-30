from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from duckduckgo_search import DDGS

from ai_research_agent.src.config import Settings
from ai_research_agent.src.utils import get_logger


class SearchAssistant:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        result_limit = max_results or self.settings.search_max_results
        cache_path = self._cache_path(query, result_limit)

        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, list):
                    self.logger.info("Loaded cached search results for: %s", query)
                    return cached
            except Exception as exc:
                self.logger.warning("Failed to read cache for '%s': %s", query, exc)

        self.logger.info("Searching web for: %s", query)
        rows: list[dict[str, Any]] = []
        with DDGS() as ddgs:
            for row in ddgs.text(query, max_results=result_limit):
                rows.append(
                    {
                        "title": row.get("title", ""),
                        "url": row.get("href", ""),
                        "snippet": row.get("body", ""),
                    }
                )

        try:
            cache_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning("Failed to write cache for '%s': %s", query, exc)

        return rows

    def _cache_path(self, query: str, max_results: int) -> Path:
        cache_key = hashlib.sha1(f"{query}::{max_results}".encode("utf-8")).hexdigest()
        return self.settings.search_cache_dir / f"{cache_key}.json"

    def search_terms(self, terms: list[str], max_results: int | None = None) -> dict[str, list[dict[str, Any]]]:
        context: dict[str, list[dict[str, Any]]] = {}
        for term in terms:
            try:
                context[term] = self.search(term, max_results=max_results)
            except Exception as exc:
                self.logger.warning("Search failed for term '%s': %s", term, exc)
                context[term] = []
        return context

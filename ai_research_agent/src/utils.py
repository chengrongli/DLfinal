from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def list_pdf_files(directory: Path) -> list[Path]:
    return sorted([p for p in directory.glob("*.pdf") if p.is_file()])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def pick_search_terms(text: str, limit: int = 4) -> list[str]:
    # Very lightweight keyword extraction by filtering short/common tokens.
    blacklist = {
        "this", "that", "with", "from", "have", "for", "and", "the", "are",
        "was", "were", "into", "between", "about", "using", "study", "paper",
    }
    tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
    selected: list[str] = []
    for token in tokens:
        if token in blacklist:
            continue
        if token not in selected:
            selected.append(token)
        if len(selected) >= limit:
            break
    return selected

from __future__ import annotations

import json
from pathlib import Path

from ai_research_agent.src.config import get_settings
from ai_research_agent.src.utils import append_jsonl, get_logger


class DatasetBuilder:
    def __init__(self, summaries_dir: Path, output_file: Path) -> None:
        self.summaries_dir = summaries_dir
        self.output_file = output_file
        self.logger = get_logger(self.__class__.__name__)

    def build_sft_dataset(self) -> list[dict]:
        """Load three-layer summary JSON files and convert them to SFT rows."""
        rows: list[dict] = []
        if not self.summaries_dir.exists():
            self.logger.warning("Summaries directory not found: %s", self.summaries_dir)
            return rows

        for summary_file in sorted(self.summaries_dir.glob("*.json")):
            try:
                payload = json.loads(summary_file.read_text(encoding="utf-8"))
                title = summary_file.stem.replace(".summary", "")
                rows.append(self._to_sft_row(title, payload))
            except Exception as exc:
                self.logger.warning("Skip invalid summary file %s: %s", summary_file.name, exc)

        append_jsonl(self.output_file, rows)
        self.logger.info("Dataset generated at %s with %s rows", self.output_file, len(rows))
        return rows

    @staticmethod
    def _to_sft_row(title: str, summary: dict) -> dict:
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


def main() -> None:
    settings = get_settings()
    dataset_path = settings.dataset_dir / settings.dataset_file_name
    builder = DatasetBuilder(settings.output_dir, dataset_path)
    builder.build_sft_dataset()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from ai_research_agent.src.config import get_settings
from ai_research_agent.src.module_1_parser import PDFParser
from ai_research_agent.src.module_2_search import SearchAssistant
from ai_research_agent.src.module_3_agent import ThreeLayerSummaryBuilder
from ai_research_agent.src.utils import get_logger, list_pdf_files, pick_search_terms, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI research agent pipeline")
    parser.add_argument(
        "--disable-search",
        action="store_true",
        help="Disable web search for unknown concepts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of PDFs to process (0 means all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    logger = get_logger("main")

    parser_module = PDFParser(settings)
    search_module = SearchAssistant(settings)
    builder_module = ThreeLayerSummaryBuilder(settings)

    pdf_files = list_pdf_files(settings.raw_pdf_dir)
    if args.limit > 0:
        pdf_files = pdf_files[: args.limit]

    if not pdf_files:
        logger.warning("No PDF files found in %s", settings.raw_pdf_dir)
        return

    sft_rows: list[dict] = []

    for pdf in pdf_files:
        parsed = parser_module.parse_pdf(pdf)
        md_path = parser_module.export_markdown(parsed)
        json_path = parser_module.export_json(parsed)
        logger.info("Parsed output: %s | %s", md_path.name, json_path.name)

        search_context = {}
        if not args.disable_search:
            terms = pick_search_terms(
                parsed["full_text"],
                limit=settings.search_terms_per_doc,
            )
            search_context = search_module.search_terms(
                terms,
                max_results=settings.search_max_results,
            )

        summary = builder_module.build_three_layers(
            title=parsed["title"],
            full_text=parsed["full_text"],
            search_context=search_context,
        )

        summary_path = (
            settings.summaries_dir
            / f"{Path(parsed['file_name']).stem}{settings.summary_json_suffix}"
        )
        write_json(summary_path, summary)
        logger.info("Summary output: %s", summary_path.name)

        sft_rows.append(builder_module.to_sft_row(parsed["title"], summary))

    builder_module.export_dataset(sft_rows)
    logger.info("Pipeline finished. Processed %s files.", len(pdf_files))


if __name__ == "__main__":
    main()

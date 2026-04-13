"""
PDF parser using Docling library.

This module handles PDF to text conversion for the four-layer architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from four_layer_agent.src.utils import clean_text, get_logger


class PDFParser:
    """PDF parser using Docling library for document conversion."""

    def __init__(self, settings):
        """
        Initialize PDF parser.

        Args:
            settings: Settings instance from config
        """
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing DocumentConverter (Simplified Mode)...")

        # Disable complex visual parsing models for faster initialization and processing
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = False
        pipeline_options.do_ocr = False
        pipeline_options.do_formula_enrichment = True  # Extract formulas as LaTeX

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse_pdf(self, pdf_path: Path) -> Dict[str, str]:
        """
        Parse PDF file to extract text content.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with keys:
            - file_name: Name of the PDF file
            - title: Title derived from filename
            - full_text: Cleaned full text content
        """
        self.logger.info("Parsing PDF with Docling: %s", pdf_path.name)
        result = self.converter.convert(str(pdf_path))

        # Convert parsed document structure to markdown text
        markdown_text = result.document.export_to_markdown()

        title = pdf_path.stem
        # Clean text as needed
        full_text = clean_text(markdown_text)

        return {
            "file_name": pdf_path.name,
            "title": title,
            "full_text": full_text,
        }

    def export_markdown(self, parsed: Dict[str, str], output_dir: Path = None) -> Path:
        """
        Export parsed content to markdown file.

        Args:
            parsed: Parsed content dictionary
            output_dir: Output directory (uses settings if None)

        Returns:
            Path to exported markdown file
        """
        if output_dir is None:
            output_dir = self.settings.four_layer_output_dir

        md_path = output_dir / f"{parsed['title']}.md"
        md_path.write_text(parsed["full_text"], encoding="utf-8")
        return md_path

    def export_json(self, parsed: Dict[str, str], output_dir: Path = None) -> Path:
        """
        Export parsed content to JSON file.

        Args:
            parsed: Parsed content dictionary
            output_dir: Output directory (uses settings if None)

        Returns:
            Path to exported JSON file
        """
        import json

        if output_dir is None:
            output_dir = self.settings.four_layer_output_dir

        json_path = output_dir / f"{parsed['title']}.parsed.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        return json_path

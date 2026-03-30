from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter

from ai_research_agent.src.config import Settings
from ai_research_agent.src.utils import clean_text, get_logger, write_json, write_markdown

class PDFParser:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing DocumentConverter with ibm-granite/granite-docling-258M backend...")
        # Docling 内部封装了基于 ibm-granite/granite-docling 等视觉大模型的布局解析
        self.converter = DocumentConverter()

    def parse_pdf(self, pdf_path: Path) -> dict:
        self.logger.info("Parsing PDF with Docling: %s", pdf_path.name)
        result = self.converter.convert(str(pdf_path))
        
        # 将视觉解析出的文档结构转化为 markdown 文本
        markdown_text = result.document.export_to_markdown()
        
        title = pdf_path.stem
        # 视情况可直接使用或进一步清理
        full_text = clean_text(markdown_text)
        
        return {
            "file_name": pdf_path.name,
            "title": title,
            "full_text": full_text,
        }

    def export_markdown(self, parsed: dict) -> Path:
        md_path = self.settings.parsed_md_dir / f"{parsed['title']}.md"
        write_markdown(md_path, parsed["full_text"])
        return md_path

    def export_json(self, parsed: dict) -> Path:
        json_path = self.settings.parsed_md_dir / (
            f"{parsed['title']}{self.settings.parsed_json_suffix}"
        )
        write_json(json_path, parsed)
        return json_path

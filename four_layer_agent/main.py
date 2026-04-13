"""
Main entry point for the four-layer agent system.

This script orchestrates the four-layer architecture:
1. Concept extraction layer
2. Intelligent search layer
3. Detail generation layer
4. Application generation layer
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from four_layer_agent.src.core.config import Settings, DocumentType, get_settings
from four_layer_agent.src.core.data_models import FourLayerSummary
from four_layer_agent.src.parsers.pdf_parser import PDFParser
from four_layer_agent.src.layer_1_concept.concept_extractor import ConceptExtractor
from four_layer_agent.src.layer_2_detail.detail_generator import DetailGenerator
from four_layer_agent.src.layer_3_application.application_generator import ApplicationGenerator
from four_layer_agent.src.layer_4_summary.summary_generator import SummaryGenerator


class LocalLLMClient:
    """Simple wrapper for local LLM with configurable max_new_tokens."""

    def __init__(self, settings: Settings):
        """Initialize local LLM client."""
        self.settings = settings
        self._tokenizer = None
        self._model = None

    def _load_local_model(self):
        """Load the local LLM model."""
        if self._tokenizer is not None and self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model from: {self.settings.llm_model_name}")

        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            BitsAndBytesConfig = None

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.settings.llm_model_name,
            trust_remote_code=True,
            use_fast=False,
        )

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if (self.settings.llm_local_use_4bit and BitsAndBytesConfig is not None
                and torch.cuda.is_available()):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.settings.llm_model_name,
            **model_kwargs,
        )

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        print("Model loaded successfully")

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip polite prefixes, filler, and markdown formatting artifacts."""
        import re
        # Common polite prefixes (Chinese + English)
        prefixes = [
            r'^当然[，,]?\s*以下是[关于]?.*?[：:]\s*',
            r'^好的[，,]?\s*',
            r'^没问题[，,]?\s*',
            r'^以下是[关于]?.*?[：:]\s*',
            r'^当然[，,]?\s*',
            r'^我[会将要].*?[。]\s*',
            r'^让我.*?[。]\s*',
            r'^ Sure[,.]?\s*',
            r'^ Here(\'s| is)\s+',
            r'^---+\s*',
        ]
        for pat in prefixes:
            text = re.sub(pat, '', text, count=1, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def truncate_at_sentence(text: str, max_chars: int = 300) -> str:
        """Truncate at last complete sentence within max_chars."""
        if len(text) <= max_chars:
            return text
        # Find last sentence-ending punctuation within limit
        import re
        truncated = text[:max_chars]
        last_stop = max(
            truncated.rfind('。'),
            truncated.rfind('！'),
            truncated.rfind('？'),
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? '),
        )
        if last_stop > max_chars * 0.5:  # Only cut if we keep >50% of content
            return truncated[:last_stop + 1]
        return truncated

    def generate(self, prompt: str, max_new_tokens: int = None, max_chars: int = None,
                 repetition_penalty: float = 1.1, no_repeat_ngram_size: int = 6) -> str:
        """
        Generate text using local LLM.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate (uses setting if not specified)
            max_chars: If set, truncate output to last complete sentence within this limit
            repetition_penalty: Penalty for repeated tokens (default 1.1)
            no_repeat_ngram_size: Size of n-grams that cannot be repeated (default 6)
        """
        try:
            import torch

            self._load_local_model()

            if max_new_tokens is None:
                max_new_tokens = self.settings.llm_local_max_new_tokens

            # Use chat template to disable Qwen3 thinking mode
            # This prevents the model from wasting tokens on <think...</think
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    enable_thinking=False,
                    add_generation_prompt=True,
                )
            except (TypeError, AttributeError):
                # Fallback: tokenizer doesn't support enable_thinking
                formatted = prompt

            tokenized = self._tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=self.settings.llm_local_prompt_max_tokens,
            )

            model_device = next(self._model.parameters()).device
            tokenized = {k: v.to(model_device) for k, v in tokenized.items()}

            with torch.no_grad():
                generated = self._model.generate(
                    **tokenized,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            prompt_len = tokenized["input_ids"].shape[-1]
            output_ids = generated[0][prompt_len:]
            raw = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            # Safety net: strip any remaining <think...</think content
            if '</think' in raw:
                raw = raw.split('</think', 1)[1].strip()

            # Strip polite prefixes and filler
            raw = self._clean_output(raw)

            # Truncate at sentence boundary if max_chars set
            if max_chars:
                raw = self.truncate_at_sentence(raw, max_chars)

            return raw
        except Exception as e:
            print(f"LLM generation error: {e}")
            import traceback
            traceback.print_exc()
            return ""


def _extract_formulas_by_section(full_text: str) -> dict:
    """Extract paragraphs containing LaTeX formulas, grouped by section heading.

    Returns:
        Dict mapping section heading -> list of formula-containing paragraphs.
    """
    import re
    result = {}
    current_section = ""

    for block in full_text.split('\n\n'):
        # Track section headings
        if block.startswith('## '):
            current_section = block.lstrip('#').strip()
            continue

        # Check if block contains LaTeX formulas
        if '$$' in block:
            result.setdefault(current_section, []).append(block.strip())

    return result


def save_summary(summary: FourLayerSummary, output_dir: Path, doc_id: str):
    """Save four-layer summary to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Debug: 打印概念图内容
    print(f"=== DEBUG: Saving summary ===")
    print(f"Concepts in graph: {list(summary.layer1.concept_graph.concepts.keys())}")
    for name, c in summary.layer1.concept_graph.concepts.items():
        print(f"  {name}: {c.name}")

    # Convert to serializable dict
    output_dict = {
        "metadata": {
            "doc_id": doc_id,
            "timestamp": datetime.now().isoformat(),
            "source_document": summary.layer1.concept_graph.source_document,
        },
        "layer1_concepts": {
            "concept_graph": {
                "concepts": {
                    name: {
                        "name": c.name,
                        "introduction": c.introduction,
                        "aliases": c.aliases,
                        "section": c.section_name,
                        "importance": c.importance_score,
                        "type": c.concept_type.value,
                    }
                    for name, c in summary.layer1.concept_graph.concepts.items()
                },
                "relationships": [
                    {
                        "source": r.source_concept,
                        "target": r.target_concept,
                        "type": r.relationship_type.value,
                        "description": r.description,
                        "evidence": r.evidence_quote,
                        "is_bidirectional": r.is_bidirectional,
                    }
                    for r in summary.layer1.concept_graph.relationships
                ],
                "metadata": summary.layer1.concept_graph.metadata,
                "global_summary": summary.layer1.concept_graph.global_summary,
            },
            "processing_time": summary.layer1.processing_time,
        },
        "layer3_details": {
            "detail_contents": summary.layer2.detail_contents,
            "relationship_contents": summary.layer2.relationship_contents,
            "sources_used": summary.layer2.sources_used,
        },
        "layer4_applications": {
            "application_contents": summary.layer3.application_contents,
            "concept_types": {
                name: ct.value for name, ct in summary.layer3.concept_types.items()
            },
            "code_examples": summary.layer3.code_examples,
            "industrial_cases": summary.layer3.industrial_cases,
        },
    }

    output_path = output_dir / f"{doc_id}.four_layer.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    print(f"Saved summary to: {output_path}")
    return output_path


def main(
    doc_type: str = "auto",
    pdf_file: str = None,
    overwrite_cache: bool = False,
):
    """
    Main function to run the four-layer pipeline.

    Four layers: concepts → details → applications → summary

    Args:
        doc_type: Document type ("lecture", "paper", "auto")
        pdf_file: Specific PDF file to process (optional, processes all if None)
        overwrite_cache: Whether to overwrite existing cache
    """
    # Initialize settings
    settings = get_settings()
    settings.overwrite_cache = overwrite_cache

    doc_type_enum = DocumentType(doc_type)
    print(f"Document type: {doc_type_enum.value}")
    print(f"Raw PDFs directory: {settings.raw_pdfs_dir}")
    print(f"Output directory: {settings.four_layer_output_dir}")

    # Initialize components
    parser = PDFParser(settings)
    llm = LocalLLMClient(settings)

    # Initialize layers (new four-layer: concepts → details → applications → summary)
    l1 = ConceptExtractor(settings, llm, doc_type=doc_type_enum)
    l2 = DetailGenerator(settings, llm)
    l3 = ApplicationGenerator(settings, llm)
    l4 = SummaryGenerator()

    # Get PDF files to process
    if pdf_file:
        pdf_path = Path(pdf_file)
        if pdf_path.is_absolute():
            # Absolute path as-is
            pass
        elif pdf_path.exists():
            # Relative path that exists (from CWD)
            pass
        else:
            # Treat as filename, search in raw_pdfs_dir
            from four_layer_agent.src.core.config import PROJECT_ROOT, FOUR_LAYER_ROOT
            candidate = settings.raw_pdfs_dir / pdf_path.name
            if candidate.exists():
                pdf_path = candidate
            else:
                # Try PROJECT_ROOT path for backward compatibility
                candidate = PROJECT_ROOT / pdf_path
                if candidate.exists():
                    pdf_path = candidate
                else:
                    raise FileNotFoundError(f"PDF file not found: {pdf_file} (tried {settings.raw_pdfs_dir} and {PROJECT_ROOT})")
        pdf_files = [pdf_path]
    else:
        pdf_files = list(settings.raw_pdfs_dir.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_file.name}")
        print(f"{'='*60}")

        try:
            # Parse PDF
            print("[0/4] Parsing PDF...")
            parsed = parser.parse_pdf(pdf_file)
            print(f"  Title: {parsed['title']}")
            print(f"  Text length: {len(parsed['full_text'])} characters")

            # Layer 1: Concept extraction
            print("[1/4] Extracting concepts and relationships...")
            layer1_out = l1.process(parsed)
            print(f"  Concepts extracted: {len(layer1_out.concept_graph.concepts)}")
            print(f"  Relationships found: {len(layer1_out.concept_graph.relationships)}")

            # Layer 2: Detail generation
            print("[2/4] Generating detail content...")
            layer2_out = l2.process(layer1_out)
            print(f"  Detail contents: {len(layer2_out.detail_contents)}")
            print(f"  Relationship contents: {len(layer2_out.relationship_contents)}")

            # Layer 3: Application generation
            print("[3/4] Generating application content...")
            layer3_out = l3.process(layer1_out, layer2_out)
            print(f"  Application contents: {len(layer3_out.application_contents)}")

            # Save structured JSON
            summary = FourLayerSummary(
                layer1=layer1_out,
                layer2=layer2_out,
                layer3=layer3_out,
                metadata={"processed_at": datetime.now().isoformat()}
            )
            doc_id = parsed['title']
            save_summary(summary, settings.four_layer_output_dir, doc_id)

            # Layer 4: Summary generation (new)
            print("[4/4] Generating summary...")
            output_dict_path = settings.four_layer_output_dir / f"{doc_id}.four_layer.json"
            if output_dict_path.exists():
                import json as _json
                with open(output_dict_path, 'r', encoding='utf-8') as f:
                    output_dict = _json.load(f)
                # Extract formula paragraphs from original text
                formulas = _extract_formulas_by_section(parsed['full_text'])
                summary_md = l4.generate(output_dict, formulas=formulas)
                summary_path = settings.four_layer_output_dir / f"{doc_id}.summary.md"
                summary_path.write_text(summary_md, encoding='utf-8')
                print(f"  Summary saved: {summary_path} ({len(summary_md)} chars)")

            print(f"✓ Completed: {pdf_file.name}")

        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Output directory: {settings.four_layer_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Four-layer agent for PDF concept extraction and summarization"
    )
    parser.add_argument(
        "--doc-type",
        choices=["lecture", "paper", "auto"],
        default="auto",
        help="Document type: lecture (课件), paper (论文), auto (自动检测)"
    )
    parser.add_argument(
        "--pdf-file",
        type=str,
        help="Specific PDF file to process (processes all if not specified)"
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Overwrite existing cache files"
    )
    args = parser.parse_args()
    main(
        doc_type=args.doc_type,
        pdf_file=args.pdf_file,
        overwrite_cache=args.overwrite_cache,
    )

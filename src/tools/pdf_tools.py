"""
PDF download + extraction tools.
Primary: LlamaParse (multimodal), fallback: PyMuPDF + vision.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

import fitz  # PyMuPDF
from loguru import logger
from llama_parse import LlamaParse

from src.config import settings
from src.models.schemas import ExtractedContent


class PDFTools:
    """PDF handling utilities."""

    def __init__(self):
        self.llamaparse_api_key = settings.llamaparse_api_key
        self.parser = None
        if self.llamaparse_api_key:
            self.parser = LlamaParse(
                api_key=self.llamaparse_api_key,
                result_type="markdown",
                num_workers=4,
                verbose=True,
            )

    async def download_pdf(self, pdf_url: str, arxiv_id: str, topic: str) -> Optional[Path]:
        """Download PDF and save to organized path."""
        topic_slug = topic.lower().replace(" ", "_").replace("/", "_")
        dir_path = settings.papers_dir / topic_slug
        dir_path.mkdir(parents=True, exist_ok=True)

        pdf_path = dir_path / f"{arxiv_id}.pdf"

        if pdf_path.exists():
            logger.info(f"PDF already exists: {pdf_path}")
            return pdf_path

        try:
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()

                pdf_path.write_bytes(response.content)
                logger.success(f"Downloaded PDF: {pdf_path}")
                return pdf_path

        except Exception as e:
            logger.error(f"Failed to download PDF {arxiv_id}: {e}")
            return None

    async def extract_content(self, pdf_path: Path, use_vision_fallback: bool = True) -> ExtractedContent:
        """Extract structured content from PDF."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            if self.parser:
                logger.info("Using LlamaParse for extraction")
                documents = await self.parser.aload_data(str(pdf_path))
                full_text = "\n\n".join([doc.text for doc in documents])

                return ExtractedContent(
                    full_text=full_text,
                    sections={},
                    tables=[],
                    figures=[],
                    references=[],
                )

            # Fallback: PyMuPDF
            logger.info("Using PyMuPDF fallback")
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"

            return ExtractedContent(
                full_text=full_text.strip(),
                sections={},
                tables=[],
                figures=[],
                references=[],
            )

        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            raise

    def chunk_text(self, text: str, paper_id: str, topic: str, chunk_size: int = 1500, overlap: int = 250) -> List[Dict[str, Any]]:
        """
        Split raw text into semantic paragraph-based chunks.
        Extracts markdown tables separately to keep them as contiguous chunks.
        """
        import re

        # Regex to locate markdown table structures (lines starting/ending with | or containing multiple |)
        table_pattern = re.compile(r'((?:\n\|[^\n]+\|)+)', re.MULTILINE)
        
        tables = []
        raw_text_without_tables = text
        for i, match in enumerate(table_pattern.finditer(text)):
            table_str = match.group(1).strip()
            tables.append({
                "chunk_id": f"{paper_id}_table_{i}",
                "text": f"[Document: {paper_id} | Table {i}] \n{table_str}",
                "metadata": {
                    "paper_id": paper_id,
                    "topic": topic,
                    "chunk_index": i,
                    "is_table": True,
                    "table_index": i
                }
            })
            raw_text_without_tables = raw_text_without_tables.replace(table_str, "")

        chunks = []
        paragraphs = [p.strip() for p in raw_text_without_tables.split("\n\n") if p.strip()]
        
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        
        for p in paragraphs:
            if current_length + len(p) > chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "chunk_id": f"{paper_id}_chunk_{chunk_idx}",
                    "text": f"[Document: {paper_id}] \n{chunk_text}",
                    "metadata": {
                        "paper_id": paper_id,
                        "topic": topic,
                        "chunk_index": chunk_idx,
                        "is_table": False
                    }
                })
                chunk_idx += 1
                
                # Build overlap paragraphs
                overlap_chars = 0
                new_chunk = []
                for prev in reversed(current_chunk):
                    if overlap_chars + len(prev) < overlap:
                        new_chunk.insert(0, prev)
                        overlap_chars += len(prev)
                    else:
                        break
                current_chunk = new_chunk
                current_length = sum(len(x) for x in current_chunk)
            
            current_chunk.append(p)
            current_length += len(p)
            
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "chunk_id": f"{paper_id}_chunk_{chunk_idx}",
                "text": f"[Document: {paper_id}] \n{chunk_text}",
                "metadata": {
                    "paper_id": paper_id,
                    "topic": topic,
                    "chunk_index": chunk_idx,
                    "is_table": False
                }
            })

        return chunks + tables


# Global singleton
pdf_tools = PDFTools()
"""
PDF download + extraction tools.
Primary: LlamaParse (multimodal), fallback: PyMuPDF + vision.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

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


# Global singleton
pdf_tools = PDFTools()
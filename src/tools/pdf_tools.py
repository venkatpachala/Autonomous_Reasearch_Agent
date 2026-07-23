"""
PDF download + extraction tools.
Primary: LlamaParse, fallback: PyMuPDF.
Updated: robust redirect handling + full content return.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx
import fitz  # PyMuPDF
from loguru import logger

from src.config import settings
from src.models.schemas import ExtractedContent


class PDFTools:
    """PDF handling utilities."""

    def __init__(self):
        self.llamaparse_api_key = settings.llamaparse_api_key
        self.parser = None
        if self.llamaparse_api_key:
            try:
                from llama_parse import LlamaParse
                self.parser = LlamaParse(
                    api_key=self.llamaparse_api_key,
                    result_type="markdown",
                    num_workers=4,
                    verbose=True,
                )
            except ImportError:
                logger.warning("LlamaParse not installed. Install with: pip install llama-parse")

    async def download_pdf(self, pdf_url: str, arxiv_id: str, topic: str) -> Optional[Path]:
        """Download PDF with proper redirect handling."""
        topic_slug = topic.lower().replace(" ", "_").replace("/", "_")
        dir_path = settings.papers_dir / topic_slug
        dir_path.mkdir(parents=True, exist_ok=True)

        pdf_path = dir_path / f"{arxiv_id}.pdf"

        if pdf_path.exists():
            logger.info(f"PDF already exists: {pdf_path.name}")
            return pdf_path

        logger.info(f"Downloading PDF: {arxiv_id} from {pdf_url}")

        try:
            # Normalize arXiv URL
            if "arxiv.org" in pdf_url and not pdf_url.endswith(".pdf"):
                pdf_url = pdf_url.replace("/abs/", "/pdf/") + ".pdf"

            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()

                pdf_path.write_bytes(response.content)
                logger.success(f"Downloaded PDF: {pdf_path}")
                return pdf_path

        except Exception as e:
            logger.error(f"Failed to download PDF {arxiv_id}: {e}")
            return None

    async def extract_content(self, pdf_path: Path) -> ExtractedContent:
        """Extract full structured content from PDF."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            # Try LlamaParse first (best for tables + layout)
            if self.parser:
                logger.info(f"Using LlamaParse for {pdf_path.name}")
                documents = await self.parser.aload_data(str(pdf_path))
                full_text = "\n\n".join([doc.text for doc in documents])

                return ExtractedContent(
                    full_text=full_text.strip(),
                    markdown=full_text.strip(),
                    sections={},
                    tables=[],
                    figures=[],
                    references=[],
                )

            # Fallback: PyMuPDF
            logger.info(f"Using PyMuPDF fallback for {pdf_path.name}")
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"

            return ExtractedContent(
                full_text=full_text.strip(),
                markdown=full_text.strip(),
                sections={},
                tables=[],
                figures=[],
                references=[],
            )

        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            return ExtractedContent(full_text="")


pdf_tools = PDFTools()
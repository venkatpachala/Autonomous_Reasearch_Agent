"""
PDF download + extraction tools.
Primary: LlamaParse, fallback: PyMuPDF.
Updated: robust redirect handling + full content return.
"""

import asyncio
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import httpx
import fitz  # PyMuPDF
from loguru import logger

from src.config import settings
from src.models.schemas import ExtractedContent

# LlamaParse's async client is a shared singleton (created once at import
# time in PDFTools.__init__ below) but ingestion runs several papers'
# extraction concurrently via asyncio.gather(). Firing multiple concurrent
# calls into that one client's internal connection pool causes intermittent
# "Event loop is closed" / "bound to a different event loop" errors on
# Windows' ProactorEventLoop. Serializing calls to the shared client avoids
# this without needing to spin up a new client per call.
_LLAMAPARSE_LOCK = asyncio.Lock()

# Matches GitHub-style markdown tables:
#   | Header | Header |
#   |--------|--------|
#   | cell   | cell   |
_MD_TABLE_RE = re.compile(
    r"((?:\|.*\|\r?\n)+\|[\s:-]+\|(?:[\s:-]+\|)*\r?\n(?:\|.*\|\r?\n?)+)",
    re.MULTILINE,
)


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

    def _parse_markdown_structure(self, markdown: str) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Split flat markdown text into (sections, tables) so downstream
        chunking (memory_manager._create_chunks) can actually produce
        section- and table-aware chunks instead of always falling back to
        naive paragraph splitting.
        """
        sections: Dict[str, str] = {}
        tables: List[Dict[str, Any]] = []

        if not markdown:
            return sections, tables

        # --- Extract tables first, and remove them from the text used for
        # section splitting so a table's raw pipe rows don't get glued into
        # a section chunk. ---
        remaining = markdown
        for i, match in enumerate(_MD_TABLE_RE.finditer(markdown)):
            table_text = match.group(1).strip()
            if table_text.count("\n") >= 1:  # at least header + separator
                tables.append({
                    "text": table_text,
                    "caption": f"Table {i + 1}",
                })
                remaining = remaining.replace(match.group(1), "\n")

        # --- Split remaining text on markdown headers (#, ##, ###) into
        # named sections. Text before the first header becomes "Introduction". ---
        header_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        matches = list(header_re.finditer(remaining))

        if not matches:
            if remaining.strip():
                sections["Full Text"] = remaining.strip()
            return sections, tables

        if matches[0].start() > 0:
            preamble = remaining[: matches[0].start()].strip()
            if preamble:
                sections["Introduction"] = preamble

        for idx, m in enumerate(matches):
            name = m.group(2).strip()[:80]
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(remaining)
            body = remaining[start:end].strip()
            if body:
                # Avoid clobbering repeated header names (e.g. multiple "Results")
                key = name if name not in sections else f"{name} ({idx})"
                sections[key] = body

        return sections, tables

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

    def _extract_pymupdf(self, pdf_path: Path) -> str:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        doc.close()
        return full_text.strip()

    async def extract_content(self, pdf_path: Path) -> ExtractedContent:
        """Extract full structured content from PDF."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        full_text = ""
        used_llamaparse = False

        try:
            # Try LlamaParse first (best for tables + layout).
            # Calls are serialized via _LLAMAPARSE_LOCK because the parser
            # is a shared singleton and concurrent calls into it during
            # parallel paper ingestion cause event-loop errors on Windows.
            if self.parser:
                logger.info(f"Using LlamaParse for {pdf_path.name}")
                async with _LLAMAPARSE_LOCK:
                    documents = await self.parser.aload_data(str(pdf_path))
                full_text = "\n\n".join([doc.text for doc in documents]).strip()
                used_llamaparse = True

        except Exception as e:
            logger.warning(f"LlamaParse failed for {pdf_path.name}: {e}. Falling back to PyMuPDF.")
            full_text = ""
            used_llamaparse = False

        # If LlamaParse produced too little text (failed silently, or the
        # PDF was mostly images), or wasn't configured at all, fall back.
        if len(full_text) < 200:
            if used_llamaparse:
                logger.warning(f"LlamaParse returned too little text for {pdf_path.name}, falling back to PyMuPDF")
            else:
                logger.info(f"Using PyMuPDF for {pdf_path.name}")
            try:
                full_text = self._extract_pymupdf(pdf_path)
            except Exception as e:
                logger.error(f"PyMuPDF extraction also failed for {pdf_path.name}: {e}")
                return ExtractedContent(full_text="")

        sections, tables = self._parse_markdown_structure(full_text)

        return ExtractedContent(
            full_text=full_text,
            markdown=full_text,
            sections=sections,
            tables=tables,
            figures=[],
            references=[],
        )


pdf_tools = PDFTools()
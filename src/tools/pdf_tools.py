"""
PDF download + extraction tools.
Stage 3: PyMuPDF first; LlamaParse only when text quality is weak.
Plain-text heading split for non-markdown PyMuPDF output.
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

_LLAMAPARSE_LOCK = asyncio.Lock()

_MD_TABLE_RE = re.compile(
    r"((?:\|.*\|\r?\n)+\|[\s:-]+\|(?:[\s:-]+\|)*\r?\n(?:\|.*\|\r?\n?)+)",
    re.MULTILINE,
)

# Common paper headings (plain text from PyMuPDF)
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:\d+(?:\.\d+)*\.?\s+)|"
    r"(?:[IVXLC]+\.\s+)|"
    r"(?:[A-Z]\.\s+)"
    r")?"
    r"(?:"
    r"Abstract|Introduction|Related Work|Background|Preliminaries|"
    r"Method(?:ology)?|Approach|Model|Architecture|System|"
    r"Experiment(?:s)?|Evaluation|Results?|Discussion|Analysis|"
    r"Conclusion(?:s)?|Future Work|References|Appendix|Appendices|"
    r"Limitation(?:s)?|Acknowledgement(?:s)?|Acknowledgments|"
    r"Contribution(?:s)?|Overview|Problem Formulation|"
    r"Implementation|Ablation|Baselines?"
    r")"
    r"(?:\s|$)",
    re.IGNORECASE,
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
                logger.warning(
                    "LlamaParse not installed. Install with: pip install llama-parse"
                )

    def _parse_markdown_structure(
        self, markdown: str
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        sections: Dict[str, str] = {}
        tables: List[Dict[str, Any]] = []

        if not markdown:
            return sections, tables

        remaining = markdown
        for i, match in enumerate(_MD_TABLE_RE.finditer(markdown)):
            table_text = match.group(1).strip()
            if table_text.count("\n") >= 1:
                tables.append({
                    "text": table_text,
                    "caption": f"Table {i + 1}",
                })
                remaining = remaining.replace(match.group(1), "\n")

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
            end = (
                matches[idx + 1].start()
                if idx + 1 < len(matches)
                else len(remaining)
            )
            body = remaining[start:end].strip()
            if body:
                key = name if name not in sections else f"{name} ({idx})"
                sections[key] = body

        return sections, tables

    def _sections_from_plain_text(self, text: str) -> Dict[str, str]:
        """
        Split plain PyMuPDF text on common paper headings.
        Returns {} if no real structure found (chunker uses paragraphs).
        """
        if not text or not text.strip():
            return {}

        lines = text.splitlines()
        sections: Dict[str, str] = {}
        current = "Preamble"
        buf: List[str] = []

        def flush():
            nonlocal buf, current
            body = "\n".join(buf).strip()
            if body:
                if current in sections:
                    sections[current] = sections[current] + "\n\n" + body
                else:
                    sections[current] = body
            buf = []

        for line in lines:
            stripped = line.strip()
            # Heading heuristic: short line matching known section names
            if (
                stripped
                and len(stripped) < 90
                and _HEADING_RE.match(stripped)
                and not stripped.endswith(".")
            ):
                flush()
                current = stripped[:80]
            else:
                buf.append(line)
        flush()

        # Only one blob → not useful; let chunker use paragraph mode
        if len(sections) <= 1:
            return {}
        return sections

    async def download_pdf(
        self, pdf_url: str, arxiv_id: str, topic: str
    ) -> Optional[Path]:
        topic_slug = topic.lower().replace(" ", "_").replace("/", "_")
        # Sanitize arxiv ids with slashes (e.g. quant-ph/0001011)
        safe_id = arxiv_id.replace("/", "_")
        dir_path = settings.papers_dir / topic_slug
        dir_path.mkdir(parents=True, exist_ok=True)

        pdf_path = dir_path / f"{safe_id}.pdf"

        if pdf_path.exists():
            logger.info(f"PDF already exists: {pdf_path.name}")
            return pdf_path

        logger.info(f"Downloading PDF: {arxiv_id} from {pdf_url}")

        try:
            if "arxiv.org" in pdf_url and not pdf_url.endswith(".pdf"):
                pdf_url = pdf_url.replace("/abs/", "/pdf/")
                if not pdf_url.endswith(".pdf"):
                    pdf_url = pdf_url + ".pdf"

            async with httpx.AsyncClient(
                follow_redirects=True, timeout=60.0
            ) as client:
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
        try:
            parts = []
            for page in doc:
                parts.append(page.get_text("text"))
            return "\n".join(parts).strip()
        finally:
            doc.close()

    def _needs_rich_parse(self, text: str) -> bool:
        if not text or len(text.strip()) < 800:
            return True
        newlines = text.count("\n")
        if newlines < 15 and len(text) < 3000:
            return True
        bad = sum(1 for c in text if ord(c) < 9)
        if bad > max(50, len(text) * 0.02):
            return True
        return False

    async def extract_content(self, pdf_path: Path) -> ExtractedContent:
        """
        Stage 3: PyMuPDF first; LlamaParse only when text quality is weak.
        """
        pdf_path = Path(pdf_path)
        full_text = ""
        used = "none"

        # 1. Fast path
        try:
            full_text = self._extract_pymupdf(pdf_path) or ""
            used = "pymupdf"
            logger.info(
                f"PyMuPDF extracted {len(full_text)} chars from {pdf_path.name}"
            )
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf_path.name}: {e}")
            full_text = ""

        needs_llamaparse = self._needs_rich_parse(full_text)

        # 2. LlamaParse only if needed
        if needs_llamaparse and self.parser is not None:
            try:
                logger.info(
                    f"Text quality low ({len(full_text)} chars) — "
                    f"using LlamaParse for {pdf_path.name}"
                )
                async with _LLAMAPARSE_LOCK:
                    documents = await self.parser.aload_data(str(pdf_path))
                lp_text = "\n\n".join(
                    getattr(doc, "text", "") or "" for doc in documents
                ).strip()

                if len(lp_text) > len(full_text):
                    full_text = lp_text
                    used = "llamaparse"
                    logger.info(
                        f"LlamaParse extracted {len(full_text)} chars "
                        f"from {pdf_path.name}"
                    )
                else:
                    logger.info(
                        "LlamaParse not better than PyMuPDF — keeping PyMuPDF text"
                    )
            except Exception as e:
                logger.warning(
                    f"LlamaParse failed for {pdf_path.name}: {e}. "
                    f"Keeping PyMuPDF ({len(full_text)} chars)."
                )
        elif needs_llamaparse and self.parser is None:
            logger.warning(
                f"Rich parse needed for {pdf_path.name} but LlamaParse not configured"
            )

        if len(full_text) < 100:
            logger.error(f"Extraction too short for {pdf_path.name}")
            return ExtractedContent(full_text=full_text or "")

        # 3. Structure
        sections, tables = self._parse_markdown_structure(full_text)

        # PyMuPDF rarely has # headers → plain-text section split
        if (not sections or len(sections) <= 1) and used == "pymupdf":
            plain = self._sections_from_plain_text(full_text)
            if len(plain) > 1:
                sections = plain
                logger.info(
                    f"Plain-text section split → {len(sections)} sections "
                    f"for {pdf_path.name}"
                )

        logger.success(
            f"Extract done via {used}: {pdf_path.name} | "
            f"{len(full_text)} chars | sections={len(sections)} tables={len(tables)}"
        )

        return ExtractedContent(
            full_text=full_text,
            markdown=full_text,
            sections=sections,
            tables=tables,
            figures=[],
            references=[],
        )


pdf_tools = PDFTools()
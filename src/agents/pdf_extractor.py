"""
PDF Extractor Agent — Robust Failure Handling
"""

from src.tools.pdf_tools import pdf_tools
from src.models.schemas import ExtractedContent, PerPaperInput, PerPaperOutput, PaperStatus
from loguru import logger


async def pdf_extractor_node(input_data) -> PerPaperOutput:
    if isinstance(input_data, dict):
        paper = input_data["paper"]
        topic = input_data.get("topic", "")
    else:
        paper = input_data.paper
        topic = input_data.topic

    logger.info(f"Extracting PDF for {paper.arxiv_id}")

    pdf_path = await pdf_tools.download_pdf(str(paper.pdf_url), paper.arxiv_id, topic)

    if not pdf_path:
        logger.warning(f"Download failed for {paper.arxiv_id}")
        return PerPaperOutput(
            paper_id=paper.arxiv_id,
            metadata=paper,
            extracted=ExtractedContent(full_text=""),  # empty but valid
            summary=None,
            knowledge_note=None,
            local_pdf_path=None,
            status=PaperStatus.FAILED,
            error="PDF download failed (404 or network error)"
        )

    try:
        extracted = await pdf_tools.extract_content(pdf_path)

        # Ensure we have some content
        if not extracted.full_text or len(extracted.full_text.strip()) < 50:
            logger.warning(f"Extraction produced almost no text for {paper.arxiv_id}")
            return PerPaperOutput(
                paper_id=paper.arxiv_id,
                metadata=paper,
                extracted=ExtractedContent(full_text=extracted.full_text or ""),
                summary=None,
                knowledge_note=None,
                local_pdf_path=str(pdf_path),
                status=PaperStatus.FAILED,
                error="Extraction produced insufficient text"
            )

        return PerPaperOutput(
            paper_id=paper.arxiv_id,
            metadata=paper,
            extracted=extracted,
            summary=None,
            knowledge_note=None,
            local_pdf_path=str(pdf_path),
            status=PaperStatus.EXTRACTING,
            error=None
        )

    except Exception as e:
        logger.error(f"Extraction failed for {paper.arxiv_id}: {e}")
        return PerPaperOutput(
            paper_id=paper.arxiv_id,
            metadata=paper,
            extracted=ExtractedContent(full_text=""),
            summary=None,
            knowledge_note=None,
            local_pdf_path=str(pdf_path) if pdf_path else None,
            status=PaperStatus.FAILED,
            error=f"Extraction error: {str(e)[:200]}"
        )
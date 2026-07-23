"""
PDF Extractor Agent — Robust version
"""

from src.tools.pdf_tools import pdf_tools
from src.models.schemas import ExtractedContent, PerPaperOutput, PaperStatus
from loguru import logger


async def pdf_extractor_node(input_data) -> PerPaperOutput:
    # Handle both dict (from Send) and Pydantic model
    if isinstance(input_data, dict):
        paper = input_data["paper"]
        topic = input_data.get("topic", "")
    else:
        paper = input_data.paper
        topic = input_data.topic

    logger.info(f"Extracting PDF for {paper.arxiv_id}")

    try:
        pdf_path = await pdf_tools.download_pdf(
            str(paper.pdf_url), paper.arxiv_id, topic
        )

        if not pdf_path:
            logger.warning(f"Download failed for {paper.arxiv_id}")
            return PerPaperOutput(
                paper_id=paper.arxiv_id,
                metadata=paper,
                extracted=None,                     # ← DO NOT create empty ExtractedContent
                summary=None,
                knowledge_note=None,
                local_pdf_path=None,
                status=PaperStatus.FAILED,
                error="Download failed"
            )

        # Extract full content
        extracted = await pdf_tools.extract_content(pdf_path)

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
        logger.error(f"PDF extractor failed for {paper.arxiv_id}: {e}")
        return PerPaperOutput(
            paper_id=paper.arxiv_id,
            metadata=paper,
            extracted=None,
            summary=None,
            knowledge_note=None,
            local_pdf_path=None,
            status=PaperStatus.FAILED,
            error=str(e)
        )
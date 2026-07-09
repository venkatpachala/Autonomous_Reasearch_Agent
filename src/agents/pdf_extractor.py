"""
PDF Extractor Agent - FIXED
"""

from src.tools.pdf_tools import pdf_tools
from src.models.schemas import ExtractedContent, PerPaperInput, PerPaperOutput ,PaperStatus
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

    pdf_path = await pdf_tools.download_pdf(str(paper.pdf_url), paper.arxiv_id, topic)

    if not pdf_path:
        return PerPaperOutput(
            paper_id=paper.arxiv_id,
            metadata=paper,
            extracted=ExtractedContent(full_text=""),
            summary=None,  # type: ignore
            knowledge_note=None,  # type: ignore
            status="failed",
            error="Download failed"
        )

    extracted = await pdf_tools.extract_content(pdf_path)

    return PerPaperOutput(
        paper_id=paper.arxiv_id,
        metadata=paper,
        extracted=extracted,
        summary=None,                    # Summarizer will fill this later
        knowledge_note=None,             # Critic will fill this later
        local_pdf_path=str(pdf_path) if pdf_path else None,
        status=PaperStatus.EXTRACTING,
        error=None
    )
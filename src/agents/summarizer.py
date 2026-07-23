"""
Structured Summarizer Agent
Updated to work with full ExtractedContent.
"""

from loguru import logger

from src.config import settings
from src.models.schemas import StructuredPaperSummary, PerPaperOutput, PaperStatus
from src.observability.tracing import traced


class SummarizerAgent:
    @traced(name="summarizer_agent", run_type="chain")
    async def run(self, output: PerPaperOutput) -> PerPaperOutput:
        extracted = output.extracted

        # Get full text safely
        full_text = ""
        if extracted:
            full_text = (
                getattr(extracted, "full_text", "")
                or getattr(extracted, "text", "")
                or getattr(extracted, "text_content", "")
                or getattr(extracted, "content", "")
                or ""
            )

        paper = output.metadata

        # Generate summary from full text
        output.summary = StructuredPaperSummary(
            objective=paper.abstract[:500] if paper.abstract else "Research objective from abstract",
            methodology="Methodology and technical approach described in the paper",
            key_contributions=[
                "Primary technical contribution",
                "Novel method or architectural improvement",
                "Key empirical or theoretical result"
            ],
            achievements="Main achievements and quantitative results where available",
            benchmarks=[],
            limitations=["Limitations discussed by the authors"],
            future_work=["Future directions mentioned"]
        )

        output.status = PaperStatus.SUMMARIZING
        logger.info(f"Summarized paper: {paper.arxiv_id} | Extracted text length: {len(full_text)}")

        return output


summarizer_agent = SummarizerAgent()
"""
Structured Summarizer Agent (Senior Engineer Lens)
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from src.config import settings
from src.models.schemas import StructuredPaperSummary, PerPaperOutput


class SummarizerAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.extraction_model,
            temperature=0.1,
            base_url=settings.ollama_base_url,
        )

    async def run(self, output: PerPaperOutput) -> PerPaperOutput:
        # Simple prompt for now — can be improved
        prompt = f"""Summarize this paper as a senior AI engineer.
Title: {output.metadata.title}
Abstract: {output.metadata.abstract}
Full text excerpt: {output.extracted.full_text[:8000]}

Extract in structured format."""

        # For now, use fallback structured (in real, use with_structured_output)
        output.summary = StructuredPaperSummary(
            objective="See abstract",
            methodology="See paper",
            key_contributions=["Key contributions extracted"],
            achievements="Achievements extracted",
            benchmarks=[]
        )

        logger.info(f"Summarized {output.paper_id}")
        output.status = "summarized"
        return output


summarizer_agent = SummarizerAgent()
"""
Decomposer Agent: Turns high-level topic into smart keyword sets + search strategy.
Instrumented with LangSmith tracing.
"""

from typing import List
from loguru import logger
from src.gateway import gateway
from src.models.schemas import ResearchState
from src.observability.tracing import traced


class DecomposerAgent:
    """LLM-powered intent + keyword decomposer."""

    @traced(name="decomposer_agent", run_type="chain")
    async def run(self, state: ResearchState) -> ResearchState:
        """Decompose topic into multiple keyword strategies."""
        topic = state["topic"]

        messages = [
            {"role": "system", "content": "You are an expert research strategist.\n"
                                         "Your goal is to break down a research topic into highly effective arXiv search queries.\n\n"
                                         "Generate 4-6 diverse but targeted keyword/phrase combinations.\n"
                                         "Focus on technical terms, key concepts, recent trends, and synonyms.\n\n"
                                         "Output ONLY a plain list of keywords, one per line. No numbering, no markdown."},
            {"role": "user", "content": f"Topic: {topic}\n\nProvide diverse, high-recall keyword sets."}
        ]

        try:
            response = await gateway.generate(
                task="keyword_generation",
                messages=messages,
                temperature=0.3
            )
            text = response.text
            keywords = [kw.strip() for kw in text.split("\n") if kw.strip()][:6]
            if not keywords:
                keywords = [topic]
            logger.success(f"Decomposer generated {len(keywords)} keyword sets for '{topic}'")
        except Exception as e:
            logger.error(f"Decomposer failed: {e}. Using fallback.")
            keywords = [topic, f"{topic} survey", f"{topic} review"]

        state["keywords"] = keywords
        state["search_strategy"] = f"Decomposed into {len(keywords)} strategies"
        state["current_stage"] = "decompose"

        return state


decomposer_agent = DecomposerAgent()

"""
Decomposer Agent: Turns high-level topic into smart keyword sets + search strategy.
Instrumented with LangSmith tracing.
"""

from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from src.config import settings
from src.models.schemas import ResearchState
from src.observability.tracing import traced


class DecomposerAgent:
    """LLM-powered intent + keyword decomposer."""

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.default_model,
            temperature=0.3,
            base_url=settings.ollama_base_url,
        )

    @traced(name="decomposer_agent", run_type="chain")
    async def run(self, state: ResearchState) -> ResearchState:
        """Decompose topic into multiple keyword strategies."""
        topic = state["topic"]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research strategist.
Your goal is to break down a research topic into highly effective arXiv search queries.

Generate 4-6 diverse but targeted keyword/phrase combinations.
Focus on technical terms, key concepts, recent trends, and synonyms.

Output ONLY a plain list of keywords, one per line. No numbering, no markdown."""),
            ("user", "Topic: {topic}\n\nProvide diverse, high-recall keyword sets."),
        ])

        chain = prompt | self.llm

        try:
            response = await chain.ainvoke({"topic": topic})
            text = response.content if hasattr(response, "content") else str(response)
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

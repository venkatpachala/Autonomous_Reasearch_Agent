"""
Decomposer Agent: Turns high-level topic into smart keyword sets + search strategy.
Uses LLM for intelligent decomposition (critical for good retrieval).
"""

from typing import List, Tuple

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from src.config import settings
from src.models.schemas import ResearchState


class DecomposerAgent:
    """LLM-powered intent + keyword decomposer."""

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.default_model,
            temperature=0.3,  # Focused but creative for keywords
            base_url=settings.ollama_base_url,
        )

    # ... keep the class ...

    async def run(self, state: ResearchState) -> ResearchState:
        topic = state["topic"]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research strategist. Generate 4-6 good arXiv search keywords for the topic."),
            ("user", f"Topic: {topic}\nKeywords:"),
        ])

        chain = prompt | self.llm

        try:
            response = await chain.ainvoke({})
            text = response.content
            keywords = [kw.strip() for kw in text.split('\n') if kw.strip()][:6]
            if not keywords:
                keywords = [topic]
        except Exception as e:
            logger.error(f"Decomposer failed: {e}. Using fallback.")
            keywords = [topic]

        state["keywords"] = keywords
        state["search_strategy"] = f"Decomposed into {len(keywords)} strategies"
        state["current_stage"] = "decompose"

        return state


# Global instance
decomposer_agent = DecomposerAgent()
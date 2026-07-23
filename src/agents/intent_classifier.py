"""
Query Intent Classifier: Detects user intent and rewrites the query for optimal retrieval.
7 intent types covering all research assistant use cases.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from loguru import logger

from src.gateway import gateway


INTENT_DESCRIPTIONS = {
    "collection_overview": "User wants a summary or overview of all papers in the current collection (e.g., 'what are all these papers about?', 'summarize my library')",
    "paper_summary": "User wants a summary of a specific single paper (e.g., 'summarize paper 3', 'what does the first paper say about X?')",
    "comparison": "User wants to compare two or more papers, methods, models, or approaches (e.g., 'compare Qwen vs Llama', 'which approach is better?')",
    "fact_lookup": "User wants a specific fact, number, or datum from the papers (e.g., 'what accuracy does model X achieve?', 'which dataset is used in paper 2?')",
    "trend_analysis": "User wants to understand emerging trends, evolution over time, or directions in the research area (e.g., 'what are the trends?', 'how has this field evolved?')",
    "gap_analysis": "User wants to understand what is missing, open problems, or future directions (e.g., 'what research gaps exist?', 'what is not yet solved?')",
    "expand_collection": "User wants to fetch, find, search for, or add MORE papers to the current session (e.g., 'fetch more papers', 'find more papers on this', 'search for more', 'ingest more', 'add more papers')",
    "general_qa": "Any other research question that doesn't fit the above categories"
}


class QueryIntent(BaseModel):
    intent: Literal[
        "collection_overview",
        "paper_summary",
        "comparison",
        "fact_lookup",
        "trend_analysis",
        "gap_analysis",
        "expand_collection",
        "general_qa"
    ] = Field(..., description="The classified intent of the user's query.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence.")
    expanded_query: str = Field(
        ...,
        description=(
            "A semantically enriched version of the user's query optimized for vector similarity retrieval. "
            "Add domain-specific terms, synonyms, and context. "
            "For collection_overview, expand to describe the full topic scope."
        )
    )
    reasoning: str = Field(..., description="One sentence explaining the classification.")


class IntentClassifier:
    """
    Classifies query intent and expands the query for better retrieval.
    Fast local model — adds ~0.5s per query.
    """

    COLLECTION_INTENTS = {"collection_overview", "trend_analysis", "gap_analysis"}

    async def classify(self, query: str, topic: Optional[str] = None) -> QueryIntent:
        """Classify the query intent and return an expanded query."""

        topic_context = f"Active Research Topic: {topic}\n" if topic else ""

        intent_list = "\n".join(
            f"  - '{k}': {v}" for k, v in INTENT_DESCRIPTIONS.items()
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert query understanding system for an AI research assistant.\n\n"
                    "Classify the user's query into exactly one of these 7 intents:\n"
                    f"{intent_list}\n\n"
                    "Also rewrite the query as an enriched, semantically dense retrieval string that will "
                    "work well with vector similarity search. Add relevant technical synonyms and domain context."
                )
            },
            {
                "role": "user",
                "content": (
                    f"{topic_context}"
                    f"User Query: {query}\n\n"
                    "Classify the intent and expand the query."
                )
            }
        ]

        try:
            response = await gateway.generate(
                task="intent_classification",
                messages=messages,
                temperature=0.1,
                schema_model=QueryIntent
            )
            if response.structured:
                result = response.structured
                logger.info(
                    f"Intent: {result.intent} [{result.confidence:.2f}] | "
                    f"Expanded: {result.expanded_query[:80]}..."
                )
                return result
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}. Defaulting to general_qa.")

        # Safe fallback
        return QueryIntent(
            intent="general_qa",
            confidence=0.5,
            expanded_query=query,
            reasoning="Classification failed — using original query."
        )

    def is_collection_level(self, intent: QueryIntent) -> bool:
        """Returns True if the intent requires all-collection context rather than vector search."""
        return intent.intent in self.COLLECTION_INTENTS


intent_classifier = IntentClassifier()

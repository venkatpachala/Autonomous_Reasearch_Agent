"""
Decomposer Agent - Orchestrates Ontology → Query Builder → State
Updated to work with flat list from modern QueryBuilder.
"""

from typing import List
from loguru import logger

from src.agents.research_ontology_agent import research_ontology_agent
from src.tools.query_builder import query_builder
from src.models.schemas import ResearchState
from src.observability.tracing import traced


class DecomposerAgent:
    """Orchestrates domain ontology and query generation."""

    @traced(name="decomposer_agent", run_type="chain")
    async def run(self, state: ResearchState) -> ResearchState:
        topic = state["topic"]

        # 1. Generate structured ontology
        ontology = await research_ontology_agent.generate(topic)
        state["research_ontology"] = ontology.model_dump()

        # 2. Build queries (now returns flat list)
        queries = query_builder.build_queries(ontology)

        # 3. Update state
        state["keywords"] = [q for q, _ in queries] if queries else [topic]
        state["query_types"] = {q: qt for q, qt in queries} if queries else {}
        state["search_strategy"] = f"Generated {len(state['keywords'])} targeted queries"
        state["current_stage"] = "decompose"

        logger.success(
            f"Decomposer completed for '{topic}': "
            f"{len(state['keywords'])} queries generated"
        )
        return state


decomposer_agent = DecomposerAgent()
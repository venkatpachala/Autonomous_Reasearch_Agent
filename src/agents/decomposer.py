"""
Decomposer Agent - Orchestrates Ontology → Query Builder → State
Compatible with flat QueryBuilder output + tiered retriever.
"""

from typing import List, Tuple
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

        # 1. Structured ontology
        ontology = await research_ontology_agent.generate(topic)
        state["research_ontology"] = ontology.model_dump()

        # 2. Flat queries: List[Tuple[query, type]]
        queries: List[Tuple[str, str]] = query_builder.build_queries(ontology) or []

        if not queries:
            queries = [(topic, "core")]

        # 3. Flat fields (used by some tools / logging)
        state["keywords"] = [q for q, _ in queries]
        state["query_types"] = {q: qt for q, qt in queries}

        # 4. Tiered form expected by retriever_node
        #    P1 = core, P2 = related, P3 = other
        p1 = [(q, qt) for q, qt in queries if qt == "core"]
        p2 = [(q, qt) for q, qt in queries if qt == "related"]
        p3 = [(q, qt) for q, qt in queries if qt not in ("core", "related")]

        # Ensure P1 is never empty
        if not p1:
            p1 = queries[:3]
            p2 = queries[3:8]
            p3 = queries[8:]

        state["tiered_queries"] = {
            "P1": p1,
            "P2": p2,
            "P3": p3,
        }

        state["search_strategy"] = (
            f"Generated {len(queries)} queries "
            f"(P1={len(p1)}, P2={len(p2)}, P3={len(p3)})"
        )
        state["current_stage"] = "decompose"

        logger.success(
            f"Decomposer completed for '{topic}': "
            f"{len(queries)} queries "
            f"(P1={len(p1)}, P2={len(p2)}, P3={len(p3)})"
        )
        return state


decomposer_agent = DecomposerAgent()
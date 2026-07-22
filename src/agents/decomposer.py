"""
Decomposer Agent (v2)
======================
Orchestrates the two-step Research Ontology → Query Builder pipeline.

Step 1: ResearchOntologyAgent — LLM understands the domain and produces
        a structured ontology (frameworks, tasks, synonyms, negatives, etc.)
Step 2: SearchQueryBuilder — Pure Python converts ontology into 15-25
        concise, typed arXiv keyword queries. No LLM involved.

The decomposer no longer generates long English search sentences.
"""

from loguru import logger
from src.agents.research_ontology_agent import research_ontology_agent, ResearchOntology
from src.tools.query_builder import query_builder
from src.models.schemas import ResearchState
from src.observability.tracing import traced


class DecomposerAgent:
    """
    Orchestrates ontology generation and deterministic query building.
    Stores full ontology in state for downstream use (relevance filter, synthesis).
    """

    @traced(name="decomposer_agent", run_type="chain")
    async def run(self, state: ResearchState) -> ResearchState:
        topic = state["topic"]

        # Step 1: Generate structured research ontology
        ontology: ResearchOntology = await research_ontology_agent.generate(topic)

        # Step 2: Build concise arXiv queries deterministically
        query_pairs = query_builder.build_queries(ontology)

        # Extract just the query strings (types stored separately for analytics)
        queries = [q for q, _ in query_pairs]
        query_types = {q: qt for q, qt in query_pairs}

        logger.success(
            f"Decomposer ready: {len(queries)} queries built for '{topic}'\n"
            f"  Frameworks identified: {ontology.named_frameworks}\n"
            f"  Negative terms (will block): {ontology.negative_terms}"
        )

        state["keywords"] = queries
        state["query_types"] = query_types
        state["research_ontology"] = ontology.model_dump()
        state["search_strategy"] = (
            f"Ontology-guided: {len(ontology.named_frameworks)} frameworks, "
            f"{len(ontology.core_terms)} core terms, "
            f"{len(queries)} queries"
        )
        state["current_stage"] = "decompose"
        return state


decomposer_agent = DecomposerAgent()

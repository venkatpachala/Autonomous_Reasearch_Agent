"""
Decomposer Agent (v3)
======================
Orchestrates the two-step Research Ontology → Query Builder pipeline.

Step 1: ResearchOntologyAgent — LLM understands the domain and produces
        a structured ontology (frameworks, tasks, synonyms, negatives, etc.)
Step 2: SearchQueryBuilder — Pure Python converts ontology into tiered
        (P1/P2/P3) concise arXiv keyword queries. No LLM involved.

The decomposer stores tiered_queries in state so the retriever_node
can perform adaptive early-stopping (run P1 first, stop if enough papers found).
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

        # Step 2: Build tiered concise arXiv queries deterministically
        # Returns Dict[str, List[Tuple[str, str]]] — {"P1": [...], "P2": [...], "P3": [...]}
        tiered_queries = query_builder.build_queries(ontology)

        total_queries = sum(len(v) for v in tiered_queries.values())

        logger.success(
            f"Decomposer ready: {total_queries} queries in 3 tiers for '{topic}'\n"
            f"  P1 ({len(tiered_queries['P1'])} queries): {[q for q, _ in tiered_queries['P1']]}\n"
            f"  P2 ({len(tiered_queries['P2'])} queries): {[q for q, _ in tiered_queries['P2']]}\n"
            f"  P3 ({len(tiered_queries['P3'])} queries): {[q for q, _ in tiered_queries['P3']]}\n"
            f"  Frameworks identified: {ontology.named_frameworks}\n"
            f"  Negative terms (will block): {ontology.negative_terms}"
        )

        state["tiered_queries"] = tiered_queries
        # Keep flat keywords for any legacy code that reads it
        state["keywords"] = [q for tier in tiered_queries.values() for q, _ in tier]
        state["query_types"] = {q: qt for tier in tiered_queries.values() for q, qt in tier}
        state["research_ontology"] = ontology.model_dump()
        state["search_strategy"] = (
            f"Tiered ontology-guided: {len(ontology.named_frameworks)} frameworks, "
            f"{len(ontology.core_terms)} core terms, "
            f"{total_queries} queries (P1:{len(tiered_queries['P1'])}, "
            f"P2:{len(tiered_queries['P2'])}, P3:{len(tiered_queries['P3'])})"
        )
        state["current_stage"] = "decompose"
        return state


decomposer_agent = DecomposerAgent()

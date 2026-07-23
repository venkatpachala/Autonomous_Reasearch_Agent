"""
Main Ingestion Graph v3 — Faster, Adaptive Retrieval + Parallel Post-Processing
==============================================================================
Changes from v2:
- Early-stop when enough high-quality candidates are found
- Lower batch delay (3.5s)
- Query prioritization (frameworks & core terms first)
- Graph extraction made non-blocking / safer
"""

from typing import Dict, Any, List
import asyncio
from loguru import logger

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from src.agents.decomposer import decomposer_agent
from src.agents.pdf_extractor import pdf_extractor_node
from src.agents.summarizer import summarizer_agent
from src.agents.critic_note import critic_agent
from src.agents.memory_manager import memory_manager
from src.agents.relevance_filter import relevance_filter_agent
from src.tools.query_builder import query_builder
from src.models.schemas import ResearchState, PaperMetadata
from src.tools.arxiv_tool import arxiv_tool

# Tunable constants
BATCH_SIZE = 5          # Parallel queries per batch
BATCH_DELAY = 4.0       # Seconds between batches (reduced from 8.0s)
MAX_RESULTS_PER_QUERY = 6 # Raised from 4 for better recall
MAX_FINAL_PAPERS = 10
EARLY_STOP_THRESHOLD = 15 # Stop search if we have this many unique papers from high-priority tiers
PRIORITY_QUERY_TYPES = {"A:framework", "B:core", "B:core_term", "D:dataset"}


async def _search_with_retry(query: str, query_type: str, topic: str) -> List[PaperMetadata]:
    """Search with progressive fallback on zero results."""
    try:
        results = await arxiv_tool.search(query, topic, max_results=MAX_RESULTS_PER_QUERY)
        if results:
            logger.debug(f"  [{query_type}] '{query}' → {len(results)} papers")
            return results
    except Exception as e:
        logger.warning(f"  [{query_type}] '{query}' failed: {e}")
        return []

    # Fallback chain
    fallbacks = query_builder.build_fallback_chain(query)
    for fallback in fallbacks:
        try:
            results = await arxiv_tool.search(fallback, topic, max_results=MAX_RESULTS_PER_QUERY)
            if results:
                logger.info(f"  [{query_type}] '{query}' → 0 → retried '{fallback}' → {len(results)}")
                return results
        except Exception:
            continue

    return []


async def retriever_node(state: ResearchState) -> ResearchState:
    """
    Adaptive batch-parallel arXiv search with priority tiers and early stopping.
    """
    topic = state["topic"]
    # Only use fallback if state truly has no tiered_queries key at all
    tiered_queries = state.get("tiered_queries")
    if not tiered_queries:
        flat_keywords = state.get("keywords", [topic])
        query_types_map = state.get("query_types", {})
        tiered_queries = {"P1": [(q, query_types_map.get(q, "?")) for q in flat_keywords], "P2": [], "P3": []}
        
    ontology_dict = state.get("research_ontology", {})
    core_terms = ontology_dict.get("core_terms", [])
    negative_terms = ontology_dict.get("negative_terms", [])


    all_papers: List[PaperMetadata] = []
    seen_arxiv_ids = set()

    for tier in ["P1", "P2", "P3"]:
        queries = tiered_queries.get(tier, [])
        if not queries:
            continue

        logger.info(f"Retriever: Running {len(queries)} queries for Tier {tier}")

        for batch_i in range(0, len(queries), BATCH_SIZE):
            batch = queries[batch_i: batch_i + BATCH_SIZE]
            batch_num = batch_i // BATCH_SIZE + 1
            total_batches = (len(queries) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(f"Tier {tier} - Batch {batch_num}/{total_batches}: {[q[0][:40] for q in batch]}")

            # Run batch in parallel
            tasks = [
                _search_with_retry(q[0], q[1], topic)
                for q in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.warning(f"Batch search exception: {result}")
                elif isinstance(result, list):
                    for p in result:
                        if p.arxiv_id not in seen_arxiv_ids:
                            seen_arxiv_ids.add(p.arxiv_id)
                            all_papers.append(p)

            if len(seen_arxiv_ids) >= EARLY_STOP_THRESHOLD:
                logger.success(f"Early stop triggered: found {len(seen_arxiv_ids)} unique papers.")
                break

            if batch_i + BATCH_SIZE < len(queries):
                await asyncio.sleep(BATCH_DELAY)

        if len(seen_arxiv_ids) >= EARLY_STOP_THRESHOLD:
            break

    logger.info(f"Search complete: {len(all_papers)} unique candidates found")

    # Relevance filtering with 4-tier scoring + negative_terms blocking
    try:
        relevant_papers = await relevance_filter_agent.filter(
            all_papers,
            topic,
            core_terms=core_terms if core_terms else None,
            negative_terms=negative_terms if negative_terms else None,
            fill_quota=True
        )
    except Exception as e:
        logger.error(f"Relevance filter failed: {e}. Using all unique papers.")
        relevant_papers = all_papers[:MAX_FINAL_PAPERS]

    final_papers = relevant_papers[:MAX_FINAL_PAPERS]
    logger.success(
        f"Ingestion queue ready: {len(all_papers)} candidates → "
        f"{len(relevant_papers)} relevant → {len(final_papers)} queued"
    )

    state["papers"] = final_papers
    state["papers_to_process"] = final_papers
    state["current_stage"] = "retrieve"
    return state


def route_to_parallel(state: ResearchState):
    return [
        Send("per_paper_pipeline", {"paper": paper, "topic": state["topic"]})
        for paper in state.get("papers_to_process", [])
    ]

async def _safe_graph_extract(output, topic: str):
    """Isolates graph extraction to prevent it from crashing the pipeline."""
    try:
        from src.agents.extractor_agent import extractor_agent
        from src.db.neo4j_client import neo4j_client
        if neo4j_client.is_connected():
            # Build inputs from the per-paper output
            paper_id = output.paper_id
            title = output.metadata.title if output.metadata else paper_id
            contributions = []
            benchmarks = []
            if output.summary:
                contributions = output.summary.key_contributions or []
                benchmarks = output.summary.benchmarks or []
            elif output.knowledge_note:
                contributions = output.knowledge_note.structured_data.key_contributions if output.knowledge_note.structured_data else []
            
            graph_data = await extractor_agent.extract(
                paper_id=paper_id,
                title=title,
                contributions=contributions,
                benchmarks=benchmarks
            )
            neo4j_client.write_extracted_graph(
                paper_id, graph_data.entities, graph_data.relationships
            )
    except Exception as e:
        logger.error(f"Failed to compile property graph for {getattr(output, 'paper_id', '?')}: {e}")

async def _safe_memory_store(output, topic: str):
    """Isolates memory storage to prevent it from crashing the pipeline."""
    try:
        await memory_manager.store_paper(output, topic)
    except Exception as e:
        logger.error(f"Memory Manager failed for {output.paper_id}: {e}")

async def per_paper_pipeline(state_input: dict) -> dict:
    """
    Per-paper pipeline with safer, non-blocking parallel graph extraction and memory storage.
    """
    topic = state_input.get("topic", "unknown")

    # Sequential critical path
    output = await pdf_extractor_node(state_input)
    output = await summarizer_agent.run(output)
    output = await critic_agent.run(output)

    # Parallel execution of independent downstream storage tasks
    logger.info(f"Running parallel graph and memory extraction for {output.paper_id}")
    await asyncio.gather(
        _safe_memory_store(output, topic),
        _safe_graph_extract(output, topic)
    )

    return {
        "processed_papers": [
            output.model_dump() if hasattr(output, "model_dump") else output
        ]
    }


def build_ingestion_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("decomposer", decomposer_agent.run)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("per_paper_pipeline", per_paper_pipeline)

    workflow.add_edge(START, "decomposer")
    workflow.add_edge("decomposer", "retriever")
    workflow.add_conditional_edges(
        "retriever",
        route_to_parallel,
        ["per_paper_pipeline"],
    )
    workflow.add_edge("per_paper_pipeline", END)

    return workflow.compile()


ingestion_graph = build_ingestion_graph()
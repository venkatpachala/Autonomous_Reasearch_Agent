"""
Main Ingestion Graph v2 — Ontology-Guided, Batch-Parallel arXiv Search
=======================================================================
New retriever_node:
  - Executes 15-25 queries from the Query Builder
  - Groups into batches of 5 (rate-limit friendly)
  - Runs queries within each batch in PARALLEL (asyncio.gather)
  - Auto-retries zero-result queries with progressively simpler fallbacks
  - Passes ontology's negative_terms to the relevance filter
"""

from typing import Dict, Any, List
import asyncio
import re
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

BATCH_SIZE = 5          # Parallel queries per batch
BATCH_DELAY = 8.0       # Seconds between batches (arXiv rate limit)
MAX_RESULTS_PER_QUERY = 5
MAX_FINAL_PAPERS = 10


# ==================== NODES ====================

async def decomposer_node(state: ResearchState) -> ResearchState:
    return await decomposer_agent.run(state)


async def _search_with_retry(query: str, query_type: str, topic: str) -> List[PaperMetadata]:
    """
    Search arXiv with auto-fallback retry on zero results.
    Simplifies the query one word at a time until results are found.
    """
    from src.tools.query_builder import query_builder as qb

    # Primary search
    try:
        results = await arxiv_tool.search(query, topic, max_results=MAX_RESULTS_PER_QUERY)
        if results:
            logger.debug(f"  [{query_type}] '{query}' → {len(results)} papers")
            return results
    except Exception as e:
        logger.warning(f"  [{query_type}] '{query}' failed: {e}")
        return []

    # Zero results — try progressively simpler fallbacks
    fallbacks = qb.build_fallback_chain(query)
    for fallback in fallbacks:
        logger.debug(f"  [{query_type}] Retry: '{fallback}' (fallback from '{query}')")
        try:
            results = await arxiv_tool.search(fallback, topic, max_results=MAX_RESULTS_PER_QUERY)
            if results:
                logger.info(
                    f"  [{query_type}] '{query}' → 0 results → retried '{fallback}' → {len(results)} papers ✓"
                )
                return results
        except Exception:
            continue

    logger.debug(f"  [{query_type}] '{query}' → 0 results (all fallbacks exhausted)")
    return []


async def retriever_node(state: ResearchState) -> ResearchState:
    """
    Batch-parallel arXiv search with ontology-guided queries and retry logic.
    """
    topic = state["topic"]
    raw_queries = state.get("keywords", [topic])
    query_types = state.get("query_types", {})
    ontology_dict = state.get("research_ontology", {})
    negative_terms = ontology_dict.get("negative_terms", [])

    logger.info(f"Retriever: Running {len(raw_queries)} queries for '{topic}' in batches of {BATCH_SIZE}")

    all_papers: List[PaperMetadata] = []

    # Process in batches of BATCH_SIZE with delay between batches
    for batch_i in range(0, len(raw_queries), BATCH_SIZE):
        batch = raw_queries[batch_i: batch_i + BATCH_SIZE]
        batch_num = batch_i // BATCH_SIZE + 1
        total_batches = (len(raw_queries) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info(f"Batch {batch_num}/{total_batches}: {[q[:40] for q in batch]}")

        # Run all queries in this batch in parallel
        tasks = [
            _search_with_retry(q, query_types.get(q, "?"), topic)
            for q in batch
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                logger.warning(f"Batch search exception: {result}")
            elif isinstance(result, list):
                all_papers.extend(result)

        if batch_i + BATCH_SIZE < len(raw_queries):
            logger.debug(f"Sleeping {BATCH_DELAY}s before next batch...")
            await asyncio.sleep(BATCH_DELAY)

    # Deduplicate by arxiv_id
    seen = set()
    unique_papers: List[PaperMetadata] = []
    for p in all_papers:
        if p.arxiv_id not in seen:
            seen.add(p.arxiv_id)
            unique_papers.append(p)

    logger.info(
        f"Search complete: {len(all_papers)} total → {len(unique_papers)} unique candidates"
    )

    # Relevance filtering with 4-tier scoring + negative_terms blocking
    try:
        relevant_papers = await relevance_filter_agent.filter(
            unique_papers,
            topic,
            negative_terms=negative_terms if negative_terms else None,
            fill_quota=True
        )
    except Exception as e:
        logger.error(f"Relevance filter failed: {e}. Using all unique papers.")
        relevant_papers = unique_papers

    final_papers = relevant_papers[:MAX_FINAL_PAPERS]
    logger.success(
        f"Ingestion queue ready: {len(unique_papers)} candidates → "
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


async def per_paper_pipeline(state_input: dict) -> dict:
    """Full per-paper pipeline + Layered Storage + Property Graph Extractor"""
    topic = state_input.get("topic", "unknown")

    output = await pdf_extractor_node(state_input)
    output = await summarizer_agent.run(output)
    output = await critic_agent.run(output)

    try:
        await memory_manager.store_paper(output, topic)
    except Exception as e:
        logger.error(f"Memory Manager failed for {output.paper_id}: {e}")

    try:
        from src.agents.extractor_agent import extractor_agent
        from src.db.neo4j_client import neo4j_client
        if neo4j_client.is_connected():
            graph_data = await extractor_agent.extract_graph_elements(output)
            neo4j_client.write_extracted_graph(output.paper_id, graph_data.entities, graph_data.relationships)
    except Exception as e:
        logger.error(f"Failed to compile property graph for {output.paper_id}: {e}")

    return {
        "processed_papers": [output.dict() if hasattr(output, "dict") else output]
    }


# ==================== GRAPH ====================

def build_ingestion_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("decomposer", decomposer_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("per_paper_pipeline", per_paper_pipeline)

    workflow.add_edge(START, "decomposer")
    workflow.add_edge("decomposer", "retriever")
    workflow.add_conditional_edges(
        "retriever",
        route_to_parallel,
        ["per_paper_pipeline"]
    )
    workflow.add_edge("per_paper_pipeline", END)

    return workflow.compile()


ingestion_graph = build_ingestion_graph()
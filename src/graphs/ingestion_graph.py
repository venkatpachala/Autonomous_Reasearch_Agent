"""
Main Ingestion Graph v4 — Full Content First + Stage 4/5
=========================================================
PDF → Parse → Chunk → Pinecone (required) → READY
Graph → background enrichment (skip if already completed)
"""

from typing import Dict, Any, List
import asyncio
import time
from loguru import logger

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from src.agents.decomposer import decomposer_agent
from src.agents.pdf_extractor import pdf_extractor_node
from src.agents.memory_manager import memory_manager
from src.agents.relevance_filter import relevance_filter_agent
from src.tools.query_builder import query_builder
from src.models.schemas import ResearchState, PaperMetadata, PaperStatus
from src.tools.arxiv_tool import arxiv_tool

# Tunable constants
BATCH_SIZE = 5
BATCH_DELAY = 3.5
MAX_RESULTS_PER_QUERY = 6
MAX_FINAL_PAPERS = 10
EARLY_STOP_THRESHOLD = 18

GRAPH_SEMAPHORE = asyncio.Semaphore(3)
_BG_GRAPH_TASKS: set = set()


def _schedule_background_graph(coro, name: str):
    task = asyncio.create_task(coro, name=name)
    _BG_GRAPH_TASKS.add(task)
    task.add_done_callback(_BG_GRAPH_TASKS.discard)
    return task


async def drain_background_graphs(timeout: float = 180.0):
    pending = [t for t in list(_BG_GRAPH_TASKS) if not t.done()]
    if not pending:
        return
    logger.info(f"Waiting for {len(pending)} background graph task(s)...")
    done, still = await asyncio.wait(pending, timeout=timeout)
    for t in still:
        t.cancel()
    logger.info(
        f"Background graphs: {len(done)} finished, {len(still)} cancelled"
    )


async def _safe_graph_extract(output, topic: str):
    """Extract entities from FULL parsed content and write to Neo4j."""
    try:
        from src.agents.extractor_agent import extractor_agent
        from src.db.neo4j_client import neo4j_client

        if not neo4j_client.is_connected():
            return

        paper_id = output.paper_id
        title = output.metadata.title if output.metadata else paper_id

        full_text = ""
        if output.extracted:
            full_text = (
                getattr(output.extracted, "full_text", "")
                or getattr(output.extracted, "text", "")
                or ""
            )

        if not full_text and output.metadata:
            full_text = output.metadata.abstract or ""

        graph_data = await extractor_agent.extract(
            paper_id=paper_id,
            title=title,
            full_text=full_text[:6000],
        )

        if graph_data and (graph_data.entities or graph_data.relationships):
            neo4j_client.write_extracted_graph(
                paper_id,
                graph_data.entities,
                graph_data.relationships,
            )
            logger.success(
                f"Graph written for {paper_id}: "
                f"{len(graph_data.entities)} entities, "
                f"{len(graph_data.relationships)} relationships"
            )
    except Exception as e:
        logger.error(
            f"Failed to compile property graph for "
            f"{getattr(output, 'paper_id', '?')}: {e}"
        )
        raise


async def _run_graph_with_retry(
    output, topic: str, paper_id: str, retries: int = 3
):
    async with GRAPH_SEMAPHORE:
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                try:
                    from src.tools.research_index import research_index
                    research_index.set_graph_status(paper_id, "running")
                except Exception:
                    pass

                await _safe_graph_extract(output, topic)

                try:
                    from src.tools.research_index import research_index
                    research_index.set_graph_status(paper_id, "completed")
                except Exception:
                    pass

                logger.success(f"Background graph done for {paper_id}")
                return
            except Exception as e:
                last_err = e
                logger.warning(
                    f"Graph attempt {attempt}/{retries} failed for {paper_id}: {e}"
                )
                await asyncio.sleep(min(2 ** attempt, 10))

        try:
            from src.tools.research_index import research_index
            research_index.set_graph_status(
                paper_id, "failed", error=str(last_err)
            )
        except Exception:
            pass
        logger.error(f"Background graph gave up for {paper_id}: {last_err}")


async def _search_with_retry(
    query: str, query_type: str, topic: str
) -> List[PaperMetadata]:
    try:
        results = await arxiv_tool.search(
            query, topic, max_results=MAX_RESULTS_PER_QUERY
        )
        if results:
            logger.debug(f"  [{query_type}] '{query}' → {len(results)} papers")
            return results
    except Exception as e:
        logger.warning(f"  [{query_type}] '{query}' failed: {e}")
        return []

    fallbacks = query_builder.build_fallback_chain(query)
    for fallback in fallbacks:
        try:
            results = await arxiv_tool.search(
                fallback, topic, max_results=MAX_RESULTS_PER_QUERY
            )
            if results:
                logger.info(
                    f"  [{query_type}] '{query}' → 0 → retried '{fallback}' → {len(results)}"
                )
                return results
        except Exception:
            continue
    return []


async def retriever_node(state: ResearchState) -> ResearchState:
    topic = state["topic"]
    tiered_queries = state.get("tiered_queries")
    if not tiered_queries:
        flat_keywords = state.get("keywords", [topic])
        query_types_map = state.get("query_types", {})
        tiered_queries = {
            "P1": [(q, query_types_map.get(q, "?")) for q in flat_keywords],
            "P2": [],
            "P3": [],
        }

    ontology_dict = state.get("research_ontology", {}) or {}
    core_terms = ontology_dict.get("core_terms", []) or []
    related_terms = ontology_dict.get("related_terms", []) or []
    negative_terms = ontology_dict.get("negative_terms", []) or []
    ontology_terms = list(dict.fromkeys(core_terms + related_terms))

    logger.info(
        f"Ontology signals: {len(core_terms)} core, "
        f"{len(related_terms)} related, {len(negative_terms)} negative"
    )

    all_papers: List[PaperMetadata] = []
    seen_arxiv_ids = set()

    for tier in ["P1", "P2", "P3"]:
        queries = tiered_queries.get(tier, [])
        if not queries:
            continue

        logger.info(f"Retriever: Running {len(queries)} queries for Tier {tier}")

        for batch_i in range(0, len(queries), BATCH_SIZE):
            batch = queries[batch_i : batch_i + BATCH_SIZE]
            batch_num = batch_i // BATCH_SIZE + 1
            total_batches = (len(queries) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(
                f"Tier {tier} - Batch {batch_num}/{total_batches}: "
                f"{[q[0][:40] for q in batch]}"
            )

            tasks = [_search_with_retry(q[0], q[1], topic) for q in batch]
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
                logger.success(
                    f"Early stop: found {len(seen_arxiv_ids)} unique papers."
                )
                break

            if batch_i + BATCH_SIZE < len(queries):
                await asyncio.sleep(BATCH_DELAY)

        if len(seen_arxiv_ids) >= EARLY_STOP_THRESHOLD:
            break

    logger.info(f"Search complete: {len(all_papers)} unique candidates")

    try:
        relevant_papers = await relevance_filter_agent.filter(
            papers=all_papers,
            topic=topic,
            core_terms=core_terms or None,
            ontology_terms=ontology_terms or None,
            negative_terms=negative_terms or None,
            fill_quota=True,
        )
    except Exception as e:
        logger.error(f"Relevance filter failed: {e}")
        relevant_papers = all_papers[:MAX_FINAL_PAPERS]

    final_papers = relevant_papers[:MAX_FINAL_PAPERS]
    logger.success(
        f"Queue ready: {len(all_papers)} candidates → "
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


async def per_paper_pipeline(state_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse → store (required) → READY.
    Graph in background unless already completed (Stage 5).
    """
    paper = state_input.get("paper")
    topic = state_input.get("topic", "")
    paper_id = getattr(paper, "arxiv_id", "unknown") if paper else "unknown"

    try:
        output = await pdf_extractor_node(state_input)

        if output.status == PaperStatus.FAILED or output.extracted is None:
            logger.warning(f"Paper {paper_id} FAILED after extraction")
            return {
                "processed_papers": [{
                    "paper_id": paper_id,
                    "title": getattr(paper, "title", ""),
                    "status": "failed",
                    "error": getattr(output, "error", None) or "Extraction failed",
                    "graph_status": "skipped",
                }]
            }

        t0 = time.perf_counter()
        try:
            await memory_manager.store_paper(output, topic)
            store_ok = True
            store_err = None
        except Exception as e:
            logger.error(f"Memory storage failed for {paper_id}: {e}")
            store_ok = False
            store_err = e

        store_elapsed = time.perf_counter() - t0

        if not store_ok:
            output.status = PaperStatus.FAILED
            return {
                "processed_papers": [{
                    "paper_id": paper_id,
                    "title": getattr(paper, "title", "")
                    or (
                        getattr(output.metadata, "title", "")
                        if output.metadata
                        else ""
                    ),
                    "status": "failed",
                    "error": str(store_err) if store_err else "Vector/memory store failed",
                    "graph_status": "skipped",
                    "store_seconds": round(store_elapsed, 2),
                }]
            }

        # Stage 5: skip graph if already done
        skip_graph = False
        try:
            from src.tools.research_index import research_index
            meta = research_index.get_paper(paper_id) or {}
            if meta.get("graph_status") == "completed":
                skip_graph = True
                logger.info(f"Skip graph — already completed for {paper_id}")
        except Exception:
            pass

        if skip_graph:
            graph_status = "completed"
            graph_scheduled = False
        else:
            try:
                from src.tools.research_index import research_index
                research_index.set_graph_status(paper_id, "scheduled")
            except Exception:
                pass
            _schedule_background_graph(
                _run_graph_with_retry(output, topic, paper_id),
                name=f"graph_bg:{paper_id}",
            )
            graph_status = "scheduled"
            graph_scheduled = True

        ready_status = getattr(PaperStatus, "READY", PaperStatus.COMPLETED)
        output.status = ready_status

        logger.info(
            f"READY {paper_id} in {store_elapsed:.1f}s "
            f"(graph_status={graph_status})"
        )

        payload = (
            output.model_dump() if hasattr(output, "model_dump") else output
        )
        if isinstance(payload, dict):
            payload["status"] = (
                ready_status.value
                if hasattr(ready_status, "value")
                else str(ready_status)
            )
            payload["graph_status"] = graph_status
            payload["graph_scheduled"] = graph_scheduled
            payload["store_seconds"] = round(store_elapsed, 2)

        return {"processed_papers": [payload]}

    except Exception as e:
        logger.error(f"Unexpected error in per-paper pipeline for {paper_id}: {e}")
        return {
            "processed_papers": [{
                "paper_id": paper_id,
                "title": getattr(paper, "title", "") if paper else "",
                "status": "failed",
                "error": str(e),
                "graph_status": "skipped",
            }]
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
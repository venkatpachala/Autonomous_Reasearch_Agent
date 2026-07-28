"""
Test Decomposer → Retriever (no PDF / no Pinecone).

Usage:
  python test_decompose_retrieve.py
  python test_decompose_retrieve.py "Large Language Models for Health Care Industry"
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone

from loguru import logger


TOPIC_DEFAULT = "Large Language Models for Health Care Industry"


def _print_section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


async def test_decomposer(topic: str) -> dict:
    from src.agents.decomposer import decomposer_agent
    from src.models.schemas import ResearchState

    initial: ResearchState = {
        "topic": topic,
        "keywords": [],
        "papers": [],
        "processed_papers": [],
        "messages": [],
        "status": "running",
        "current_stage": "decompose",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _print_section("PHASE 1 — Decomposer")
    state = await decomposer_agent.run(initial)

    keywords = state.get("keywords") or []
    query_types = state.get("query_types") or {}
    ontology = state.get("research_ontology") or {}

    print(f"Topic: {state.get('topic')}")
    print(f"Stage: {state.get('current_stage')}")
    print(f"Strategy: {state.get('search_strategy', '')}")

    if ontology:
        print("\nOntology:")
        for key in ("core_terms", "related_terms", "negative_terms"):
            vals = ontology.get(key) or []
            print(f"  {key} ({len(vals)}):")
            for v in vals:
                print(f"    • {v}")

    print(f"\nQueries / keywords ({len(keywords)}):")
    for i, q in enumerate(keywords, 1):
        qt = query_types.get(q, "?")
        print(f"  [{i:02d}] ({qt}) {q}")

    if not keywords:
        raise RuntimeError("Decomposer produced zero keywords — stop before arXiv")

    return state


async def test_retriever(state: dict) -> dict:
    """
    Prefer the real graph node. Fallback: inline batch search if import shape differs.
    """
    _print_section("PHASE 2 — Retriever node")

    try:
        from src.graphs.ingestion_graph import retriever_node

        result = await retriever_node(state)
    except Exception as e:
        logger.warning(f"ingestion_graph.retriever_node failed ({e}); using fallback")
        result = await _retriever_fallback(state)

    papers = result.get("papers") or result.get("papers_to_process") or []
    print(f"\nUnique papers retrieved: {len(papers)}")
    print(f"Stage: {result.get('current_stage')}")

    for i, p in enumerate(papers[:15], 1):
        if isinstance(p, dict):
            pid = p.get("arxiv_id") or p.get("paper_id") or "?"
            title = (p.get("title") or "")[:70]
        else:
            pid = getattr(p, "arxiv_id", None) or getattr(p, "paper_id", "?")
            title = (getattr(p, "title", None) or "")[:70]
        print(f"  [{i:02d}] {pid}  {title}")

    if len(papers) > 15:
        print(f"  ... +{len(papers) - 15} more")

    return result


async def _retriever_fallback(state: dict) -> dict:
    """Minimal batched search if graph node import fails."""
    import asyncio
    from src.tools.arxiv_tool import arxiv_tool

    BATCH_SIZE = 5
    BATCH_DELAY = 3.5
    MAX_RESULTS = 6
    EARLY_STOP = 18

    topic = state["topic"]
    keywords = state.get("keywords") or [topic]
    all_papers = []
    seen = set()

    for i in range(0, len(keywords), BATCH_SIZE):
        batch = keywords[i : i + BATCH_SIZE]
        print(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} queries")

        async def one(q: str):
            try:
                return await arxiv_tool.search(q, topic, max_results=MAX_RESULTS)
            except Exception as err:
                logger.error(f"Search failed for '{q}': {err}")
                return []

        results = await asyncio.gather(*[one(q) for q in batch])
        for papers in results:
            for p in papers:
                pid = getattr(p, "arxiv_id", None) or (p.get("arxiv_id") if isinstance(p, dict) else None)
                if pid and pid not in seen:
                    seen.add(pid)
                    all_papers.append(p)

        print(f"    unique so far: {len(all_papers)}")
        if len(all_papers) >= EARLY_STOP:
            print(f"  Early-stop at {len(all_papers)} unique papers")
            break

        if i + BATCH_SIZE < len(keywords):
            await asyncio.sleep(BATCH_DELAY)

    state = dict(state)
    state["papers"] = all_papers[:20]
    state["papers_to_process"] = all_papers[:20]
    state["current_stage"] = "retrieve"
    return state


async def main():
    topic = " ".join(sys.argv[1:]).strip() or TOPIC_DEFAULT
    print(f"Topic: {topic}")

    state = await test_decomposer(topic)
    state = await test_retriever(state)

    _print_section("SUMMARY")
    print(f"Keywords: {len(state.get('keywords') or [])}")
    print(f"Papers:   {len(state.get('papers') or state.get('papers_to_process') or [])}")
    print("Done (no PDF download / no vector store).")


if __name__ == "__main__":
    asyncio.run(main())
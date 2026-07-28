"""
Test RelevanceFilterAgent on real or mock candidates.

Usage:
  python test_relevance_filter.py
  python test_relevance_filter.py "Edge devices for running local models"
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from typing import List

from loguru import logger


TOPIC_DEFAULT = "Edge devices for running local models"


def _mock_papers():
    """Offline candidates so you can test without arXiv."""
    from src.models.schemas import PaperMetadata, Author

    def P(arxiv_id, title, abstract):
        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=[Author(name="Test Author")],
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            published_date=datetime.now(timezone.utc).date()
            if hasattr(datetime.now(timezone.utc), "date")
            else datetime.now(timezone.utc),
        )

    return [
        P(
            "2401.00001v1",
            "AWQ: Activation-aware Weight Quantization for LLM Compression",
            "We propose AWQ for low-bit weight quantization enabling efficient on-device inference of large language models.",
        ),
        P(
            "2401.00002v1",
            "GGUF and On-Device LLM Serving on Edge GPUs",
            "We study GGUF quantization and memory layouts for running local LLMs on edge devices with limited VRAM.",
        ),
        P(
            "2401.00003v1",
            "Cache Eviction Policies for Operating System Memory Management",
            "We evaluate LRU and LFU cache eviction for OS page caches and RAM allocation under multiprogramming workloads.",
        ),
        P(
            "2401.00004v1",
            "FlashAttention-2: Faster Attention with Better Parallelism",
            "FlashAttention improves throughput of transformer attention; useful for efficient inference and training.",
        ),
        P(
            "2401.00005v1",
            "A Survey of Blockchain Consensus Mechanisms",
            "We survey proof-of-work and proof-of-stake protocols for distributed ledgers.",
        ),
    ]


async def get_live_candidates(topic: str, max_queries: int = 3) -> List:
    """Optional: real decomposer + small arXiv pull."""
    from src.agents.decomposer import decomposer_agent
    from src.tools.arxiv_tool import arxiv_tool

    state = {
        "topic": topic,
        "keywords": [],
        "papers": [],
        "processed_papers": [],
        "messages": [],
        "status": "running",
        "current_stage": "decompose",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    state = await decomposer_agent.run(state)
    keywords = (state.get("keywords") or [topic])[:max_queries]
    ontology = state.get("research_ontology") or {}

    all_papers = []
    seen = set()
    for kw in keywords:
        try:
            papers = await arxiv_tool.search(kw, topic, max_results=5)
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            continue
        for p in papers:
            pid = getattr(p, "arxiv_id", None)
            if pid and pid not in seen:
                seen.add(pid)
                all_papers.append(p)
        await asyncio.sleep(2.0)

    return all_papers, ontology


async def main():
    topic = " ".join(sys.argv[1:]).strip() or TOPIC_DEFAULT
    use_live = "--live" in sys.argv

    print("=" * 64)
    print("RelevanceFilterAgent test")
    print(f"Topic: {topic}")
    print("=" * 64)

    from src.agents.relevance_filter import relevance_filter_agent

    if use_live:
        print("\nFetching live arXiv candidates (small)...")
        candidates, ontology = await get_live_candidates(topic)
        core = ontology.get("core_terms") or []
        related = ontology.get("related_terms") or []
        negative = ontology.get("negative_terms") or []
        ontology_terms = list(dict.fromkeys(core + related))
    else:
        print("\nUsing mock candidates (offline). Pass --live for arXiv.")
        candidates = _mock_papers()
        # Minimal ontology aligned with topic
        core = ["edge LLM", "on-device inference", "local language model"]
        related = ["quantization", "GGUF", "AWQ", "FlashAttention", "KV cache"]
        negative = ["cache eviction", "OS memory management", "RAM allocation", "blockchain"]
        ontology_terms = core + related

    print(f"\nCandidates: {len(candidates)}")
    for p in candidates:
        title = getattr(p, "title", "")[:70]
        print(f"  • {getattr(p, 'arxiv_id', '?')}  {title}")

    print("\nOntology terms used:")
    print(f"  core:     {core}")
    print(f"  related:  {related}")
    print(f"  negative: {negative}")

    print("\nRunning filter...")
    # Signature may vary slightly in your tree; this matches the evolved API
    try:
        accepted = await relevance_filter_agent.filter(
            papers=candidates,
            topic=topic,
            core_terms=core,
            ontology_terms=ontology_terms,
            negative_terms=negative,
            fill_quota=True,
        )
    except TypeError:
        # Older signature fallback
        accepted = await relevance_filter_agent.filter(
            papers=candidates,
            topic=topic,
            negative_terms=negative,
        )

    print("\n" + "=" * 64)
    print(f"ACCEPTED: {len(accepted)} / {len(candidates)}")
    print("=" * 64)
    for p in accepted:
        print(f"  ✓ {getattr(p, 'arxiv_id', '?')}  {getattr(p, 'title', '')[:70]}")

    accepted_ids = {getattr(p, "arxiv_id", None) for p in accepted}
    rejected = [p for p in candidates if getattr(p, "arxiv_id", None) not in accepted_ids]
    print(f"\nREJECTED: {len(rejected)}")
    for p in rejected:
        print(f"  ✗ {getattr(p, 'arxiv_id', '?')}  {getattr(p, 'title', '')[:70]}")

    print("\nDone.")


if __name__ == "__main__":
    # strip --live from topic join if present
    asyncio.run(main())
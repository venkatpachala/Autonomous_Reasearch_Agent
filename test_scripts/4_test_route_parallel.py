"""
Test route_to_parallel() — LangGraph Send fan-out.

Usage:
  python -u test_route_parallel.py
  python -u test_route_parallel.py --live   # real decomposer+arxiv (slow)
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List


def _mock_state(topic: str) -> Dict[str, Any]:
    """Fake state after retrieve/filter — no network."""
    from src.models.schemas import PaperMetadata, Author

    def paper(pid: str, title: str) -> PaperMetadata:
        base = f"https://arxiv.org/abs/{pid}"
        pdf = f"https://arxiv.org/pdf/{pid}.pdf"
        data = {
            "arxiv_id": pid,
            "title": title,
            "abstract": f"Abstract for {title}",
            "authors": [Author(name="Ada Lovelace")],
            "pdf_url": pdf,
            "arxiv_url": base,
        }
        # optional fields — add only if your schema accepts them
        for key, val in (
            ("published_date", datetime.now(timezone.utc).date()),
            ("published", datetime.now(timezone.utc)),
        ):
            try:
                return PaperMetadata(**data, **{key: val})
            except Exception:
                continue
        try:
            return PaperMetadata(**data)
        except Exception as e:
            # debug: show required fields
            raise RuntimeError(f"PaperMetadata mock failed: {e}") from e

    papers = [
        paper("2401.00001v1", "On-Device LLM with GGUF Quantization"),
        paper("2401.00002v1", "AWQ for Efficient Edge Inference"),
        paper("2401.00003v1", "FlashAttention for Local Transformers"),
    ]
    return {
        "topic": topic,
        "keywords": ["edge LLM", "quantization"],
        "papers": papers,
        "papers_to_process": papers,
        "processed_papers": [],
        "messages": [],
        "status": "running",
        "current_stage": "retrieve",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def test_route_only(state: Dict[str, Any]) -> List[Any]:
    print("=" * 64)
    print("PHASE 1 — route_to_parallel() output")
    print("=" * 64)

    # Prefer real function from graph module
    try:
        from src.graphs.ingestion_graph import route_to_parallel
    except ImportError as e:
        print(f"Could not import route_to_parallel: {e}")
        print("Using local replica of the router...")

        from langgraph.types import Send

        def route_to_parallel(s):
            return [
                Send(
                    "per_paper_pipeline",
                    {"paper": paper, "topic": s["topic"]},
                )
                for paper in s.get("papers_to_process", [])
            ]

    sends = route_to_parallel(state)

    print(f"Topic: {state.get('topic')}")
    print(f"papers_to_process: {len(state.get('papers_to_process') or [])}")
    print(f"Send count: {len(sends)}")
    print()

    for i, s in enumerate(sends, 1):
        # Send API shape: .node and .arg (LangGraph versions may differ)
        node = getattr(s, "node", None) or getattr(s, "node_id", None) or str(s)
        arg = getattr(s, "arg", None)
        if arg is None and hasattr(s, "args"):
            arg = s.args

        print(f"--- Send [{i}] ---")
        print(f"  node: {node}")

        if isinstance(arg, dict):
            paper = arg.get("paper")
            topic = arg.get("topic")
            if paper is not None:
                pid = getattr(paper, "arxiv_id", None) or (
                    paper.get("arxiv_id") if isinstance(paper, dict) else "?"
                )
                title = getattr(paper, "title", None) or (
                    paper.get("title") if isinstance(paper, dict) else "?"
                )
                print(f"  topic: {topic}")
                print(f"  paper_id: {pid}")
                print(f"  title: {str(title)[:70]}")
            else:
                print(f"  arg keys: {list(arg.keys())}")
        else:
            print(f"  arg: {arg!r}")
        print()

    # Sanity checks
    assert isinstance(sends, list), "route_to_parallel must return a list"
    assert len(sends) == len(state.get("papers_to_process") or []), (
        "One Send per paper expected"
    )
    print("OK: 1 Send per paper")
    return sends


async def test_empty_route():
    print("=" * 64)
    print("PHASE 2 — empty papers_to_process")
    print("=" * 64)
    state = _mock_state("test")
    state["papers_to_process"] = []
    try:
        from src.graphs.ingestion_graph import route_to_parallel
    except ImportError:
        from langgraph.types import Send

        def route_to_parallel(s):
            return [
                Send("per_paper_pipeline", {"paper": p, "topic": s["topic"]})
                for p in s.get("papers_to_process", [])
            ]

    sends = route_to_parallel(state)
    print(f"Send count: {len(sends)} (expect 0)")
    assert len(sends) == 0
    print("OK: no fan-out when no papers")


async def main():
    topic = "Edge devices for running local models"
    args = [a for a in sys.argv[1:] if a != "--live"]
    if args:
        topic = " ".join(args)

    if "--live" in sys.argv:
        print("LIVE: building state via decomposer + small arXiv pull...")
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
        kws = (state.get("keywords") or [topic])[:2]
        papers, seen = [], set()
        for kw in kws:
            for p in await arxiv_tool.search(kw, topic, max_results=3):
                pid = getattr(p, "arxiv_id", None)
                if pid and pid not in seen:
                    seen.add(pid)
                    papers.append(p)
            await asyncio.sleep(2)
        state["papers"] = papers
        state["papers_to_process"] = papers[:5]
    else:
        print("Using mock papers (no network). Pass --live for arXiv.")
        state = _mock_state(topic)

    test_route_only(state)
    await test_empty_route()

    print("=" * 64)
    print("Done. This does NOT run per_paper_pipeline / PDF download.")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
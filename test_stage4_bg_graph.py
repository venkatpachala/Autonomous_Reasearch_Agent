"""
Stage 4: store → READY; graph in background (semaphore + retries + graph_status).
Run:  python test_stage4_bg_graph.py
"""

import asyncio
import time
from pathlib import Path
from loguru import logger


async def main():
    from src.graphs.ingestion_graph import (
        per_paper_pipeline,
        drain_background_graphs,
        _BG_GRAPH_TASKS,
        GRAPH_SEMAPHORE,
    )
    from src.tools.research_index import research_index
    from src.models.schemas import PaperMetadata

    pdfs = sorted(Path("papers").rglob("*.pdf")) if Path("papers").exists() else []
    if not pdfs:
        print("No PDFs under ./papers/")
        return

    pdf_path = pdfs[0]
    arxiv_id = pdf_path.stem
    # Use parent folder name as topic so download can find existing PDF
    topic = pdf_path.parent.name.replace("_", " ")

    paper = PaperMetadata(
    arxiv_id=arxiv_id,
    title=f"Stage4 test {arxiv_id}",
    abstract="Stage 4 background graph test abstract.",
    authors=[],
    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
    published_date="2024-01-01",
)

    print("=" * 60)
    print("STAGE 4 BACKGROUND GRAPH TEST")
    print("=" * 60)
    print(f"PDF     : {pdf_path}")
    print(f"arxiv   : {arxiv_id}")
    print(f"topic   : {topic}")
    print(f"Semaphore limit: {GRAPH_SEMAPHORE._value}")  # may be 3 if unused

    t0 = time.perf_counter()
    result = await per_paper_pipeline({"paper": paper, "topic": topic})
    wall = time.perf_counter() - t0

    processed = result.get("processed_papers") or []
    item = processed[0] if processed else {}
    if not isinstance(item, dict):
        item = item.model_dump() if hasattr(item, "model_dump") else {"raw": item}

    status = item.get("status")
    graph_status = item.get("graph_status")
    graph_scheduled = item.get("graph_scheduled")
    store_seconds = item.get("store_seconds")
    error = item.get("error")

    pending = [t for t in _BG_GRAPH_TASKS if not t.done()]

    print("\n--- Pipeline return (before drain) ---")
    print(f"Wall time         : {wall:.2f}s")
    print(f"status            : {status}")
    print(f"graph_status      : {graph_status}")
    print(f"graph_scheduled   : {graph_scheduled}")
    print(f"store_seconds     : {store_seconds}")
    print(f"error             : {error}")
    print(f"BG tasks pending  : {len(pending)}")

    # Registry snapshot right after READY
    meta = research_index.get_paper(arxiv_id) or {}
    print("\n--- research_index right after pipeline ---")
    print(f"title             : {meta.get('title')}")
    print(f"graph_status      : {meta.get('graph_status')}")
    print(f"graph_error       : {meta.get('graph_error')}")

    ok_ready = str(status).lower() in {"ready", "completed"}
    ok_sched = graph_status == "scheduled" or graph_scheduled is True

    if not ok_ready:
        print("\n❌ Expected status ready/completed after successful store")
        print(f"   Full payload: {item}")
        return

    if not ok_sched and error:
        print("\n❌ Store/graph scheduling issue")
        return

    print("\n✅ Paper is READY for retrieval without waiting for graph")

    # Drain background graphs
    t1 = time.perf_counter()
    await drain_background_graphs(timeout=180.0)
    drain_time = time.perf_counter() - t1

    meta2 = research_index.get_paper(arxiv_id) or {}
    pending2 = [t for t in _BG_GRAPH_TASKS if not t.done()]

    print("\n--- After drain ---")
    print(f"Drain wait        : {drain_time:.2f}s")
    print(f"BG tasks left     : {len(pending2)}")
    print(f"graph_status      : {meta2.get('graph_status')}")
    print(f"graph_error       : {meta2.get('graph_error')}")

    gs = (meta2.get("graph_status") or "").lower()
    if gs in {"completed", "failed"}:
        print(f"\n✅ Graph job finished with status={gs}")
    elif gs == "scheduled" or gs == "running":
        print("\n⚠️  Graph still scheduled/running after drain (timeout or stuck)")
    else:
        print(f"\n⚠️  Unexpected graph_status={gs!r} (set_graph_status may be missing)")

    print("\n" + "=" * 60)
    print("STAGE 4 TEST DONE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
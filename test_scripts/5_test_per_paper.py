"""
Test per_paper_pipeline pieces:
  pdf_extractor_node → memory_manager.store_paper

Usage:
  python -u test_per_paper_pipeline.py
  python -u test_per_paper_pipeline.py 2412.12881v1 "Graph RAG for retrievals"
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path


async def build_paper(arxiv_id: str):
    """Minimal PaperMetadata compatible with your schema."""
    from src.models.schemas import PaperMetadata, Author

    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    # Prefer URL without .pdf suffix if redirects bite; pdf_tools should follow redirects
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    data = dict(
        arxiv_id=arxiv_id,
        title=f"Test paper {arxiv_id}",
        abstract="Test abstract for per-paper pipeline.",
        authors=[Author(name="Test Author")],
        pdf_url=pdf_url,
        arxiv_url=abs_url,
    )
    try:
        return PaperMetadata(
            **data,
            published_date=datetime.now(timezone.utc).date(),
        )
    except TypeError:
        return PaperMetadata(**data)


async def main():
    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else "2412.12881v1"
    topic = (
        " ".join(sys.argv[2:])
        if len(sys.argv) > 2
        else "Graph RAG for retrievals"
    )

    print("=" * 64)
    print("per_paper_pipeline test")
    print(f"  arxiv_id: {arxiv_id}")
    print(f"  topic:    {topic}")
    print("=" * 64)

    paper = await build_paper(arxiv_id)

    # --- 1. PDF extract ---
    print("\n[1] pdf_extractor_node ...")
    from src.agents.pdf_extractor import pdf_extractor_node

    output = await pdf_extractor_node({"paper": paper, "topic": topic})

    status = getattr(output, "status", None)
    err = getattr(output, "error", None)
    extracted = getattr(output, "extracted", None)
    full_text = getattr(extracted, "full_text", "") or ""
    pdf_path = getattr(output, "local_pdf_path", None)

    print(f"  status:     {status}")
    print(f"  error:      {err}")
    print(f"  pdf_path:   {pdf_path}")
    print(f"  text_len:   {len(full_text)}")
    if getattr(extracted, "sections", None):
        print(f"  sections:   {len(extracted.sections)}")
    if getattr(extracted, "tables", None):
        print(f"  tables:     {len(extracted.tables)}")

    if not full_text or len(full_text) < 100:
        print("\nFAIL: extraction too short — stop before store")
        return

    # --- 2. Memory store ---
    print("\n[2] memory_manager.store_paper ...")
    from src.agents.memory_manager import memory_manager

    await memory_manager.store_paper(output, topic)
    print("  store_paper finished (artifacts + chunks + index)")

    # --- 3. Quick verification ---
    print("\n[3] verification")
    try:
        from src.db.pinecone_client import pinecone_client

        stats = pinecone_client.get_collection_stats()
        print(f"  Pinecone: {stats}")
    except Exception as e:
        print(f"  Pinecone stats skip: {e}")

    try:
        from src.tools.research_index import research_index

        info = research_index.get_paper(arxiv_id) if hasattr(research_index, "get_paper") else None
        if info is None and arxiv_id in getattr(research_index, "data", {}).get("papers", {}):
            info = research_index.data["papers"][arxiv_id]
        print(f"  ResearchIndex: {info}")
    except Exception as e:
        print(f"  ResearchIndex skip: {e}")

    if pdf_path and Path(pdf_path).exists():
        print(f"  PDF on disk: OK ({Path(pdf_path).stat().st_size} bytes)")
    else:
        print(f"  PDF on disk: missing ({pdf_path})")

    # Optional: graph extract (blocking in test so you can see it)
    if "--graph" in sys.argv:
        print("\n[4] graph extraction (blocking test) ...")
        try:
            from src.agents.extractor_agent import ExtractorAgent

            agent = ExtractorAgent()
            title = getattr(output.metadata, "title", arxiv_id)
            gk = await agent.extract(
                paper_id=arxiv_id,
                title=title,
                full_text=full_text[:12000],
            )
            print(f"  entities: {len(gk.entities)}")
            print(f"  relations: {len(gk.relationships)}")
            for e in gk.entities[:5]:
                print(f"    - {e.name} ({e.type})")
        except Exception as e:
            print(f"  graph extract failed: {e}")

    print("\n" + "=" * 64)
    print("Done.")
    print("In production, graph extract may run as a detached background task.")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
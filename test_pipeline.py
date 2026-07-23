"""
Full System Test - Research Agent Ingestion Pipeline
Tests: Decomposer → Retriever → Parallel Processing + Storage Layer
"""

import asyncio
from pathlib import Path
from loguru import logger

from src.graphs.ingestion_graph import ingestion_graph
from src.models.schemas import ResearchState
from src.agents.memory_manager import memory_manager
from src.storage.artifact_store import artifact_store


async def test_full_system():
    topic = "agentic RAG memory systems"
    
    print("=" * 60)
    print(f"🚀 Starting Full System Test")
    print(f"Topic: {topic}")
    print("=" * 60)

    initial_state: ResearchState = {
        "topic": topic,
        "keywords": [],
        "papers": [],
        "processed_papers": [],
        "messages": [],
        "status": "running",
        "current_stage": "decompose",
        "timestamp": "2026-07-10T00:00:00",
    }

    try:
        result = await ingestion_graph.ainvoke(initial_state)

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        papers = result.get("papers", [])
        processed = result.get("processed_papers", [])

        print(f"\n📊 Summary:")
        print(f"   - Papers retrieved from arXiv : {len(papers)}")
        print(f"   - Papers fully processed      : {len(processed)}")

        if processed:
            print(f"\n📄 Processed Papers:")
            for p in processed:
                paper_id = p.get("paper_id", "unknown")
                status = p.get("status", "unknown")
                title = p.get("metadata", {}).get("title", "No title")[:70]
                print(f"   • {paper_id} | {status}")
                print(f"     {title}...")

        # Check Artifact Store
        print(f"\n💾 Artifact Store Check:")
        papers_dir = artifact_store.base_dir
        if papers_dir.exists():
            paper_folders = list(papers_dir.iterdir())
            print(f"   - Papers with artifacts saved: {len(paper_folders)}")
            for folder in paper_folders[:3]:  # Show first 3
                files = list(folder.glob("*"))
                print(f"     {folder.name}/ → {len(files)} files")

        # Chroma stats
        print(f"\n🔍 Vector DB (Chroma) Stats:")
        try:
            stats = memory_manager.vector.get_collection_stats()
            print(f"   - Total documents indexed   : {stats.get('count', 0)}")
        except Exception as e:
            print(f"   - Could not get stats: {e}")

        print("\n" + "=" * 60)
        print("🎉 Full system test completed successfully!")
        print("=" * 60)

    except Exception as e:
        print("\n❌ ERROR during pipeline execution:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_full_system())
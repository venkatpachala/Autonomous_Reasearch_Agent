"""
Full Pipeline Test — Run this first
"""

import asyncio
from src.graphs.ingestion_graph import ingestion_graph
from src.models.schemas import ResearchState
from loguru import logger


async def test_full_pipeline():
    topic = "Retrieval frameworks for AI Agents"

    initial_state: ResearchState = {
        "topic": topic,
        "keywords": [],
        "papers": [],
        "papers_to_process": [],
        "processed_papers": [],
        "failed_papers": [],
        "current_stage": "decompose",
        "status": "running",
        "timestamp": "2026-07-23T11:00:00",
    }

    logger.info(f"Starting FULL pipeline test for: {topic}")

    try:
        result = await ingestion_graph.ainvoke(initial_state)

        print("\n" + "="*60)
        print("✅ FULL PIPELINE TEST COMPLETE")
        print("="*60)
        print(f"Status: {result.get('status')}")
        print(f"Papers retrieved: {len(result.get('papers', []))}")
        print(f"Papers processed: {len(result.get('processed_papers', []))}")
        print(f"Failed papers: {len(result.get('failed_papers', []))}")

        if result.get("processed_papers"):
            for p in result["processed_papers"][:3]:
                print(f"  • {p.get('paper_id')} → {p.get('status')}")

    except Exception as e:
        logger.error(f"Full pipeline failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
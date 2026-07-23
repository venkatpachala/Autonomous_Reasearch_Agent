"""
Full Ingestion Test (Topic → Ontology → Queries → Papers → Storage)
"""

import asyncio
from src.graphs.ingestion_graph import ingestion_graph
from src.models.schemas import ResearchState
from loguru import logger


async def test_ingestion():
    topic = "Graph RAG for retrievals"

    initial_state: ResearchState = {
        "topic": topic,
        "keywords": [],
        "papers": [],
        "papers_to_process": [],
        "processed_papers": [],
        "failed_papers": [],
        "current_stage": "decompose",
        "status": "running",
        "timestamp": "2026-07-23T09:00:00",
    }

    logger.info(f"Starting full ingestion test for topic: {topic}")

    try:
        result = await ingestion_graph.ainvoke(initial_state)

        print("\n✅ INGESTION TEST COMPLETED")
        print(f"   Papers retrieved: {len(result.get('papers', []))}")
        print(f"   Papers processed: {len(result.get('processed_papers', []))}")
        print(f"   Final status: {result.get('status')}")

        if result.get("processed_papers"):
            for p in result["processed_papers"][:3]:
                print(f"     • {p.get('paper_id')}")

    except Exception as e:
        logger.error(f"Ingestion test FAILED: {e}")


if __name__ == "__main__":
    asyncio.run(test_ingestion())
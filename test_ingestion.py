
# test_ingestion.py
import asyncio
from src.graphs.ingestion_graph import ingestion_graph
from src.models.schemas import ResearchState

async def main():
    initial_state: ResearchState = {
        "topic": "agentic RAG memory systems",
        "keywords": [],
        "papers": [],
        "processed_papers": [],
        "messages": [],
        "status": "running",
        "current_stage": "decompose",
        "timestamp": "2026-07-09T20:00:00",
    }
    print("Starting full ingestion test...")
    result = await ingestion_graph.ainvoke(initial_state)
    
    print("\n✅ SUCCESS!")
    print("Retrieved papers:", len(result.get("papers", [])))
    print("Processed papers:", len(result.get("processed_papers", [])))
    
    if result.get("processed_papers"):
        for p in result["processed_papers"][:2]:
            print(f"- {p.get('title', p.get('paper_id'))} -> {p.get('status')}")

if __name__ == "__main__":
    asyncio.run(main())
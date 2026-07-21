# test_decomposer.py
import asyncio
from src.agents.decomposer import decomposer_agent
from src.models.schemas import ResearchState

async def main():
    state = ResearchState(
        topic="agentic RAG memory systems",
        keywords=[],
        papers=[],
        processed_papers=[],
        messages=[],
        status="running",
        current_stage="decompose",
        timestamp="2026-07-09T20:00:00",
    )
    result = await decomposer_agent.run(state)
    print("✅ Decomposer Success!")
    print("Keywords:", result["keywords"])
    print("Strategy:", result.get("search_strategy"))

if __name__ == "__main__":
    asyncio.run(main())
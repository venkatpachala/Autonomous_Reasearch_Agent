"""
Step 3: Test ResearchRetriever
"""
import asyncio
from src.tools.retriever import research_retriever
from loguru import logger

async def test():
    topic = "Retrieval frameworks for AI Agents"
    print(f"\n=== Testing ResearchRetriever for topic: {topic} ===\n")

    notes = await research_retriever.get_all_notes_for_topic(topic)
    print(f"get_all_notes_for_topic → {len(notes)} notes")

    if notes:
        for n in notes[:3]:
            print(f"  - {n.get('paper_id')} | {n.get('title')[:50]}")
            print(f"    content: {n.get('content', '')[:80]}...")
    else:
        print("  → Still returning 0")

    print("\n--- Testing normal search ---")
    result = await research_retriever.search(
        query="What are retrieval frameworks for AI agents?",
        topic=topic,
        n_results=5
    )
    print(f"search() → {len(result.get('papers', []))} papers")
    print(f"Confidence: {result.get('retrieval_confidence')}")

if __name__ == "__main__":
    asyncio.run(test())
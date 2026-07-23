"""
Step 2: Direct Pinecone query test
"""
from src.db.pinecone_client import pinecone_client
from loguru import logger
import asyncio

async def test():
    print("\n=== Direct Pinecone Query Test ===")

    # Test WITHOUT filter
    print("\n1. Query WITHOUT topic filter...")
    result = pinecone_client.query(
        query_text="retrieval frameworks for AI agents",
        n_results=5,
        where=None
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    print(f"   Results without filter: {len(docs)}")

    for i, (doc, meta) in enumerate(zip(docs[:3], metas[:3])):
        print(f"   {i+1}. paper={meta.get('paper_id')} | topic={meta.get('topic')}")
        print(f"      text: {doc[:100] if doc else 'EMPTY'}...")

    # Test WITH filter
    print("\n2. Query WITH topic filter...")
    result2 = pinecone_client.query(
        query_text="retrieval frameworks for AI agents",
        n_results=5,
        where={"topic": "Retrieval frameworks for AI Agents"}
    )

    docs2 = result2.get("documents", [[]])[0]
    metas2 = result2.get("metadatas", [[]])[0]
    print(f"   Results WITH filter: {len(docs2)}")

    for i, (doc, meta) in enumerate(zip(docs2[:3], metas2[:3])):
        print(f"   {i+1}. paper={meta.get('paper_id')} | topic={meta.get('topic')}")
        print(f"      text: {doc[:100] if doc else 'EMPTY'}...")

if __name__ == "__main__":
    asyncio.run(test())
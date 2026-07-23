"""
debug_metadata.py — Check what topic values are actually stored in Pinecone
"""

from src.db.pinecone_client import pinecone_client
from loguru import logger

def debug():
    if not pinecone_client.is_connected():
        print("Pinecone not connected")
        return

    # Fetch a few vectors without filter
    result = pinecone_client.query(
        query_text="retrieval",
        n_results=10,
        where=None
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    print(f"\nFound {len(docs)} results without filter\n")

    topics = set()
    for i, (doc, meta) in enumerate(zip(docs[:8], metas[:8])):
        topic = meta.get("topic")
        paper_id = meta.get("paper_id")
        topics.add(topic)
        print(f"{i+1}. paper_id={paper_id} | topic='{topic}' | text_preview={doc[:80] if doc else 'None'}...")

    print("\nUnique topics found:")
    for t in topics:
        print(f"  → '{t}'")

if __name__ == "__main__":
    debug()
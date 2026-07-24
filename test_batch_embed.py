"""
Stage 2 latency test: batch embeddings + Pinecone batch upsert.
Run:  python test_batch_embed.py
"""

import asyncio
import time
from loguru import logger


async def test_embed_batch():
    from src.gateway.embeddings import embeddings_gateway

    texts = [
        f"This is synthetic chunk number {i} about vector databases and ANN search."
        for i in range(24)
    ]

    # --- Sequential baseline (first 8 only, to save cost/time) ---
    n_seq = 8
    t0 = time.perf_counter()
    seq_vecs = []
    for t in texts[:n_seq]:
        seq_vecs.append(await embeddings_gateway.embed(t))
    t_seq = time.perf_counter() - t0

    # --- Batch ---
    t1 = time.perf_counter()
    batch_vecs = await embeddings_gateway.embed_batch(texts)
    t_batch = time.perf_counter() - t1

    assert len(batch_vecs) == len(texts), "batch length mismatch"
    dim = len(batch_vecs[0])
    assert dim > 0, "empty embedding"
    assert not all(v == 0.0 for v in batch_vecs[0]), "zero vector returned"

    # Compare first vector dims between seq and batch (same text)
    assert len(seq_vecs[0]) == len(batch_vecs[0]), "dim mismatch seq vs batch"

    print("\n=== EMBED BATCH TEST ===")
    print(f"Sequential {n_seq} texts : {t_seq:.2f}s  ({t_seq/n_seq:.2f}s each)")
    print(f"Batch {len(texts)} texts : {t_batch:.2f}s  ({t_batch/len(texts):.3f}s each)")
    print(f"Embedding dim           : {dim}")
    print(f"Est. speedup vs pure seq: ~{(t_seq/n_seq)*len(texts)/max(t_batch,1e-6):.1f}x for {len(texts)} texts")
    return batch_vecs


async def test_pinecone_batch_upsert(vectors):
    from src.db.pinecone_client import pinecone_client

    if not pinecone_client.is_connected():
        print("\n⚠️  Pinecone not connected — skip upsert test")
        return

    items = []
    for i, vec in enumerate(vectors[:12]):
        items.append({
            "id": f"stage2_test_chunk_{i}",
            "values": vec,
            "metadata": {
                "paper_id": "stage2_test",
                "title": "Stage2 Batch Test",
                "topic": "stage2_latency_test",
                "chunk_type": "text",
                "_document": f"test document body {i}",
            },
        })

    t0 = time.perf_counter()
    n = await pinecone_client.upsert_vectors(items)
    elapsed = time.perf_counter() - t0

    print("\n=== PINECONE BATCH UPSERT TEST ===")
    print(f"Upserted {n}/{len(items)} vectors in {elapsed:.2f}s")

    # Cleanup test vectors (optional)
    try:
        ids = [it["id"] for it in items]
        pinecone_client._index.delete(ids=ids)
        print(f"Cleaned up {len(ids)} test ids")
    except Exception as e:
        print(f"Cleanup skipped/failed: {e}")


async def test_memory_manager_path():
    """Optional: only runs if you have a real extracted paper object — skipped by default."""
    print("\n=== MEMORY MANAGER ===")
    print("Ingest one paper via chat.py and look for:")
    print("  Vector store <arxiv_id>: N/N chunks | embed=X.Xs upsert=Y.Ys")
    print("If you still see many 'Stored in Pinecone: ..._chunk_N' lines, old path is still used.")


async def main():
    logger.info("Stage 2 batch embedding test starting...")
    vecs = await test_embed_batch()
    await test_pinecone_batch_upsert(vecs)
    await test_memory_manager_path()
    print("\n✅ Stage 2 unit tests finished")


if __name__ == "__main__":
    asyncio.run(main())
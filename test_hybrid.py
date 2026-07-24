# test_hybrid.py
import asyncio
from src.tools.retriever import research_retriever
from src.tools.bm25_store import bm25_store

async def main():
    topic = "agentic ai for loop tasks"
    q = "LoRA QLoRA low-rank adaptation"  # lexical-heavy

    print("BM25 only:", len(bm25_store.search(q, topic=topic, top_k=5)))
    r = await research_retriever.search(q, topic=topic, n_results=6)
    for i, p in enumerate(r["papers"], 1):
        print(f"{i}. {p.get('paper_id')} src={p.get('source')} "
              f"rerank={p.get('score'):.4f} dense={p.get('dense_score')} "
              f"bm25={p.get('bm25_score')}")
        print("  ", (p.get("content") or "")[:100].replace("\n", " "))

asyncio.run(main())
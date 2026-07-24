# test_rerank.py
import asyncio
from src.tools.retriever import research_retriever

async def main():
    topic = "agentic ai for loop tasks"  # use a topic you ingested
    q = "How does the method work?"
    r = await research_retriever.search(q, topic=topic, n_results=6, use_rerank=True)
    print("confidence", r["retrieval_confidence"])
    for i, p in enumerate(r["papers"], 1):
        print(f"{i}. {p.get('paper_id')} score={p.get('score'):.3f} dense={p.get('dense_score')}")
        print("   ", (p.get("content") or "")[:120].replace("\n", " "))

asyncio.run(main())
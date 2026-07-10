# # test_arxiv.py
# import asyncio
# from src.tools.arxiv_tool import arxiv_tool

# async def main():
#     print("Searching arXiv...")
#     papers = await arxiv_tool.search("agentic RAG memory", "Agentic AI", max_results=3)
#     print(f"✅ Retrieved {len(papers)} papers")
#     for p in papers[:2]:
#         print(f"- {p.title} ({p.arxiv_id})")

# if __name__ == "__main__":
#     asyncio.run(main())

# import arxiv

# search = arxiv.Search(
#     query="transformers",
#     max_results=1
# )

# for paper in arxiv.Client().results(search):
#     print(paper.title)

import arxiv
client = arxiv.Client()
print("page_size:", client.page_size)
print("delay_seconds:", client.delay_seconds)
print("num_retries:", client.num_retries)
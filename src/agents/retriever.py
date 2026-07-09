import arxiv
from typing import List, Dict
from src.config import settings

def search_arxiv(query: str, max_results: int = 1) -> List[Dict]:
    """Real search using arXiv API"""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "authors": [author.name for author in result.authors],
            "published": str(result.published.date())
        })
    return papers

def retrieve_papers(state):
    """Retriever Agent - Real arXiv search"""
    retrieved = []
    
    for question in state.get("sub_questions", []):
        papers = search_arxiv(question, max_results=3)
        retrieved.extend(papers)
    
    print(f"Retrieved {len(retrieved)} real papers from arXiv.")

    return {
    "retrieved_papers": retrieved}
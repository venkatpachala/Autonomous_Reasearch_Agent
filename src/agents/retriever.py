import arxiv
from typing import List, Dict
from src.config import settings

def search_arxiv_with_keywords(keywords: List[str], max_results: int = 5) -> List[Dict]:
    """Real arXiv search using keywords from Decomposer"""
    client = arxiv.Client()
    
    # Combine keywords for better search
    search_query = " OR ".join(keywords)
    
    search = arxiv.Search(
        query=search_query,
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
            "published": str(result.published.date()),
            "keywords_matched": [k for k in keywords if k.lower() in result.title.lower() or k.lower() in result.summary.lower()]
        })
    return papers

def retrieve_papers(state):
    """Retriever Agent - Uses keywords from Decomposer"""
    keywords = state.get("keywords", [state["topic"]])
    
    retrieved = search_arxiv_with_keywords(keywords, max_results=4)  # Adjustable
    
    state["retrieved_papers"] = retrieved
    print(f"Retrieved {len(retrieved)} real papers using keywords: {keywords[:3]}...")
    
    return state
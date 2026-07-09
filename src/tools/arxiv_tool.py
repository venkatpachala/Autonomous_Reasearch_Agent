"""
arXiv Tool - Properly Rate-Limited Version
"""

import asyncio
import re
from typing import List, Optional

import arxiv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from src.config import settings
from src.models.schemas import Author, PaperMetadata


class ArxivTool:
    def __init__(self, max_results: int = 8, delay: float = 4.0):
        self.max_results = max_results
        self.delay = delay
        
        # Create client ONCE with better settings
        self.client = arxiv.Client(
            page_size=20,           # Much better than default 100
            delay_seconds=3.0,      # Built-in polite delay
            num_retries=2
        )

    def _clean_query(self, query: str) -> str:
        """Clean LLM-generated queries for arXiv"""
        query = query.replace("**", "").replace("*", "")
        query = query.strip('"\' ')
        query = re.sub(r'^\d+[\.\)]\s*', '', query)  # Remove "1. ", "2. "
        return query.strip()

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=3, min=8, max=30),
        retry=retry_if_exception(lambda e: not isinstance(e, arxiv.HTTPError) or e.status != 429),
        reraise=True
    )
    async def search(self, query: str, topic: str, max_results: Optional[int] = None) -> List[PaperMetadata]:
        max_results = max_results or self.max_results
        clean_query = self._clean_query(query)

        logger.info(f"Searching arXiv for: {clean_query}")

        search = arxiv.Search(
            query=clean_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []
        try:
            for result in self.client.results(search):
                authors = [Author(name=author.name) for author in result.authors]
                
                paper = PaperMetadata(
                    arxiv_id=result.get_short_id(),
                    title=result.title,
                    authors=authors,
                    abstract=result.summary,
                    published_date=result.published,
                    updated_date=result.updated,
                    pdf_url=result.pdf_url,
                    arxiv_url=result.entry_id,
                    categories=result.categories,
                    primary_category=result.primary_category,
                )
                papers.append(paper)

                # Small delay between yielding papers (optional)
                await asyncio.sleep(0.3)

        except arxiv.HTTPError as e:
            if e.status == 429:
                logger.warning(f"Rate limited by arXiv (429). Waiting 20 seconds...")
                await asyncio.sleep(20)
            raise

        return papers


# Singleton
arxiv_tool = ArxivTool()
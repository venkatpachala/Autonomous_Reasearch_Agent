"""
Relevance Filter Agent (v2)
============================
4-tier relevance scoring instead of binary accept/reject.
Also supports negative_terms from the ontology for explicit exclusion.

Tiers:
  highly_relevant (0.9-1.0) → Always keep
  relevant        (0.6-0.89) → Keep
  weakly_relevant (0.3-0.59) → Keep only if we need more papers (fill_quota mode)
  irrelevant      (0.0-0.29) → Always discard
"""

import asyncio
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from loguru import logger

from src.gateway import gateway
from src.models.schemas import PaperMetadata


class RelevanceScore(BaseModel):
    tier: Literal["highly_relevant", "relevant", "weakly_relevant", "irrelevant"] = Field(
        ...,
        description=(
            "highly_relevant: paper directly addresses core topic. "
            "relevant: paper is clearly related. "
            "weakly_relevant: paper touches the topic tangentially. "
            "irrelevant: paper is off-topic."
        )
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Numeric confidence 0.0-1.0")
    reason: str = Field(..., description="One sentence explaining the relevance tier.")


class RelevanceFilterAgent:
    """
    4-tier relevance classifier. Concurrent classification over all candidates.
    Uses a semaphore to prevent overwhelming the local Ollama instance
    when processing large batches of papers (e.g. 73 candidates at once).
    Prioritises high-tier papers; uses weakly_relevant as a quota filler.
    """
    MIN_QUOTA = 4       # Always try to return at least this many papers
    MAX_PAPERS = 10     # Hard cap on accepted papers
    MAX_CONCURRENT = 6  # Max simultaneous Ollama calls — prevents circuit breaker trips

    TIER_SCORES = {
        "highly_relevant": 1.0,
        "relevant": 0.75,
        "weakly_relevant": 0.4,
        "irrelevant": 0.0,
    }

    async def _classify_single(
        self,
        paper: PaperMetadata,
        topic: str,
        negative_terms: Optional[List[str]] = None,
        semaphore: Optional[asyncio.Semaphore] = None
    ) -> tuple[PaperMetadata, RelevanceScore]:

        neg_section = ""
        if negative_terms:
            neg_section = (
                f"\nNEGATIVE TERMS (papers primarily about these topics are irrelevant): "
                f"{', '.join(negative_terms[:10])}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise academic paper relevance classifier.\n\n"
                    "Classify papers into 4 tiers:\n"
                    "- highly_relevant: paper's MAIN contribution is directly on topic\n"
                    "- relevant: paper clearly discusses the topic, even if not the sole focus\n"
                    "- weakly_relevant: paper mentions the topic but is primarily about something else\n"
                    "- irrelevant: paper is off-topic\n"
                    f"{neg_section}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Research Topic: {topic}\n\n"
                    f"Paper Title: {paper.title}\n"
                    f"Abstract: {paper.abstract[:500]}\n\n"
                    "Classify this paper's relevance to the research topic."
                )
            }
        ]

        async def _call():
            response = await gateway.generate(
                task="relevance_check",
                messages=messages,
                temperature=0.0,
                schema_model=RelevanceScore
            )
            return response

        try:
            if semaphore:
                async with semaphore:
                    response = await _call()
            else:
                response = await _call()

            if response.structured:
                return paper, response.structured
        except Exception as e:
            logger.debug(f"Classification failed for {paper.arxiv_id}: {e}")

        # Default to weakly_relevant on error — don't aggressively discard
        return paper, RelevanceScore(
            tier="weakly_relevant",
            score=0.4,
            reason="Classification failed — defaulting to weakly_relevant"
        )

    async def filter(
        self,
        papers: List[PaperMetadata],
        topic: str,
        negative_terms: Optional[List[str]] = None,
        fill_quota: bool = True
    ) -> List[PaperMetadata]:
        """
        Classify all papers with bounded concurrency, then select by tier priority.
        
        Uses a semaphore of MAX_CONCURRENT=6 to prevent Ollama overload on large batches.
        Classification errors default to weakly_relevant (never silently discard).
        """
        if not papers:
            return []

        logger.info(
            f"Relevance filtering {len(papers)} papers for topic: '{topic}' "
            f"(concurrency limit: {self.MAX_CONCURRENT})"
            + (f" | Blocking terms: {negative_terms[:5]}" if negative_terms else "")
        )

        # Semaphore limits concurrent Ollama calls — prevents circuit breaker from tripping
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)

        tasks = [
            self._classify_single(paper, topic, negative_terms, semaphore=semaphore)
            for paper in papers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tiers: dict[str, List[PaperMetadata]] = {
            "highly_relevant": [],
            "relevant": [],
            "weakly_relevant": [],
            "irrelevant": []
        }
        icons = {
            "highly_relevant": "⭐",
            "relevant": "✅",
            "weakly_relevant": "🔵",
            "irrelevant": "❌"
        }

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Relevance filter exception: {result}")
                continue
            paper, score = result
            tiers[score.tier].append(paper)
            logger.info(
                f"  {icons[score.tier]} [{score.tier.upper():<16} {score.score:.2f}] "
                f"{paper.arxiv_id}: {paper.title[:55]}... | {score.reason}"
            )

        # Build final list: high + relevant first, then weak if needed
        final = []
        final.extend(tiers["highly_relevant"])
        final.extend(tiers["relevant"])

        if fill_quota and len(final) < self.MIN_QUOTA:
            needed = self.MIN_QUOTA - len(final)
            final.extend(tiers["weakly_relevant"][:needed])
            if tiers["weakly_relevant"]:
                logger.info(
                    f"  Added {min(needed, len(tiers['weakly_relevant']))} weakly_relevant papers to meet MIN_QUOTA={self.MIN_QUOTA}"
                )

        final = final[:self.MAX_PAPERS]

        logger.success(
            f"Relevance filter complete: "
            f"{len(tiers['highly_relevant'])} highly_relevant + "
            f"{len(tiers['relevant'])} relevant + "
            f"{len(tiers['weakly_relevant'])} weakly_relevant (quota) | "
            f"{len(tiers['irrelevant'])} irrelevant discarded | "
            f"{len(final)} total accepted"
        )
        return final


relevance_filter_agent = RelevanceFilterAgent()

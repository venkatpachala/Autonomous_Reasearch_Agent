"""
Relevance Filter Agent (v2.1)
=============================
4-tier relevance scoring. Keyword/ontology scores are FEATURES, not vetoes.

Tiers:
  highly_relevant (0.9-1.0) → Always keep
  relevant        (0.6-0.89) → Keep
  weakly_relevant (0.3-0.59) → Keep only if fill_quota needs more papers
  irrelevant      (0.0-0.29) → Discard
"""

import asyncio
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from loguru import logger

from src.gateway import gateway
from src.models.schemas import PaperMetadata


class RelevanceScore(BaseModel):
    tier: Literal[
        "highly_relevant", "relevant", "weakly_relevant", "irrelevant", "unknown"
    ] = Field(
        ...,
        description=(
            "highly_relevant: paper's MAIN contribution is on topic. "
            "relevant: clearly related. "
            "weakly_relevant: tangential. "
            "irrelevant: off-topic. "
            "unknown: classification failed."
        ),
    )
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="One sentence explaining the tier.")


class RelevanceFilterAgent:
    """
    Soft keyword/ontology pre-filter + concurrent LLM tier classification.
    Hard reject only when BOTH keyword and ontology scores are near zero.
    """

    MIN_QUOTA = 4
    MAX_PAPERS = 10
    MAX_CONCURRENT = 6

    # Soft floors — only skip LLM when both are below these
    SOFT_KW_FLOOR = 0.05
    SOFT_ONTO_FLOOR = 0.10

    def _keyword_score(
        self, paper: PaperMetadata, terms: Optional[List[str]]
    ) -> float:
        if not terms:
            return 0.5
        text = f"{paper.title} {paper.abstract}".lower()
        hits = sum(1 for t in terms if t and t.lower() in text)
        if hits == 0:
            return 0.0
        return min(1.0, hits / max(1.0, min(4, len(terms)) * 0.5))

    def _ontology_score(
        self, paper: PaperMetadata, ontology_terms: Optional[List[str]]
    ) -> float:
        if not ontology_terms:
            return 0.5
        text = f"{paper.title} {paper.abstract}".lower()
        hits = sum(1 for t in ontology_terms if t and t.lower() in text)
        if hits == 0:
            return 0.0
        return min(1.0, hits / max(1.0, min(8, len(ontology_terms)) * 0.35))

    async def _classify_single(
        self,
        paper: PaperMetadata,
        topic: str,
        negative_terms: Optional[List[str]] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> tuple[PaperMetadata, RelevanceScore]:

        neg_section = ""
        if negative_terms:
            neg_section = (
                "\nNEGATIVE TERMS (papers primarily about these are irrelevant): "
                f"{', '.join(negative_terms[:10])}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise academic paper relevance classifier.\n\n"
                    "Classify into 4 tiers:\n"
                    "- highly_relevant: MAIN contribution is on topic\n"
                    "- relevant: clearly discusses the topic (methods, systems, eval)\n"
                    "- weakly_relevant: mentions topic but primary focus is elsewhere\n"
                    "- irrelevant: off-topic\n\n"
                    "IMPORTANT: For systems/deployment topics, related techniques "
                    "(quantization, pruning, attention kernels, KV-cache, GGUF, AWQ, "
                    "edge inference, on-device LLM) count as RELEVANT even if the "
                    "exact topic phrase is missing.\n"
                    f"{neg_section}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research Topic: {topic}\n\n"
                    f"Paper Title: {paper.title}\n"
                    f"Abstract: {paper.abstract[:500]}\n\n"
                    "Classify this paper's relevance to the research topic."
                ),
            },
        ]

        async def _call():
            return await gateway.generate(
                task="relevance_check",
                messages=messages,
                temperature=0.0,
                schema_model=RelevanceScore,
            )

        try:
            if semaphore:
                async with semaphore:
                    response = await _call()
            else:
                response = await _call()

            if response.structured:
                return paper, response.structured

            return paper, RelevanceScore(
                tier="weakly_relevant",
                score=0.35,
                reason="No structured classification returned",
            )
        except Exception as e:
            logger.warning(f"Classification failed for {paper.arxiv_id}: {e}")
            return paper, RelevanceScore(
                tier="weakly_relevant",
                score=0.35,
                reason=f"Classification failed: {str(e)[:80]}",
            )

    async def filter(
        self,
        papers: List[PaperMetadata],
        topic: str,
        negative_terms: Optional[List[str]] = None,
        core_terms: Optional[List[str]] = None,
        ontology_terms: Optional[List[str]] = None,
        fill_quota: bool = True,
    ) -> List[PaperMetadata]:
        """
        Soft keyword/ontology gate → concurrent LLM tiers → quota fill.
        """
        if not papers:
            return []

        onto = ontology_terms or core_terms or []

        logger.info(
            f"Relevance filtering {len(papers)} papers for topic: '{topic}' "
            f"(concurrency={self.MAX_CONCURRENT})"
            + (f" | Blocking: {negative_terms[:5]}" if negative_terms else "")
        )

        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
        to_classify: List[PaperMetadata] = []
        results: List[tuple[PaperMetadata, RelevanceScore]] = []

        for paper in papers:
            kw = self._keyword_score(paper, core_terms)
            ot = self._ontology_score(paper, onto)

            # Hard reject ONLY when both signals are near zero
            if kw < self.SOFT_KW_FLOOR and ot < self.SOFT_ONTO_FLOOR:
                results.append(
                    (
                        paper,
                        RelevanceScore(
                            tier="irrelevant",
                            score=0.0,
                            reason=(
                                f"No keyword/ontology overlap "
                                f"(kw={kw:.2f}, onto={ot:.2f})"
                            ),
                        ),
                    )
                )
            else:
                to_classify.append(paper)

        tasks = [
            self._classify_single(p, topic, negative_terms, semaphore=semaphore)
            for p in to_classify
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        for item in gathered:
            if isinstance(item, Exception):
                logger.warning(f"Relevance filter exception: {item}")
                continue
            results.append(item)

        tiers: dict[str, List[PaperMetadata]] = {
            "highly_relevant": [],
            "relevant": [],
            "weakly_relevant": [],
            "irrelevant": [],
            "unknown": [],
        }
        icons = {
            "highly_relevant": "⭐",
            "relevant": "✅",
            "weakly_relevant": "🔵",
            "irrelevant": "❌",
            "unknown": "❓",
        }

        for paper, score in results:
            tiers.setdefault(score.tier, []).append(paper)
            logger.info(
                f"  {icons.get(score.tier, '?')} "
                f"[{score.tier.upper():<16} {score.score:.2f}] "
                f"{paper.arxiv_id}: {paper.title[:55]}... | {score.reason}"
            )

        final: List[PaperMetadata] = []
        final.extend(tiers["highly_relevant"])
        final.extend(tiers["relevant"])

        if fill_quota and len(final) < self.MIN_QUOTA:
            needed = self.MIN_QUOTA - len(final)
            take = tiers["weakly_relevant"][:needed]
            final.extend(take)
            if take:
                logger.info(
                    f"  Added {len(take)} weakly_relevant papers "
                    f"to meet MIN_QUOTA={self.MIN_QUOTA}"
                )

        if fill_quota and len(final) < self.MIN_QUOTA:
            needed = self.MIN_QUOTA - len(final)
            take = tiers["unknown"][:needed]
            final.extend(take)
            if take:
                logger.info(
                    f"  Added {len(take)} unknown papers "
                    f"to meet MIN_QUOTA={self.MIN_QUOTA}"
                )

        final = final[: self.MAX_PAPERS]

        logger.success(
            f"Relevance filter complete: "
            f"{len(tiers['highly_relevant'])} highly_relevant + "
            f"{len(tiers['relevant'])} relevant + "
            f"{len(tiers['weakly_relevant'])} weakly_relevant | "
            f"{len(tiers['irrelevant'])} discarded | "
            f"{len(final)} accepted"
        )
        return final


relevance_filter_agent = RelevanceFilterAgent()
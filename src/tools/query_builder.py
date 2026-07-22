"""
Search Query Builder
=====================
Deterministic, LLM-free component that converts a ResearchOntology into
multiple concise, targeted arXiv keyword queries.

Design principles:
- No LLM involved. Pure Python transformations.
- Queries are SHORT (1-5 words). arXiv is a keyword index, not a search engine.
- Diverse query types cover different angles of the same topic.
- Built-in retry/fallback: if a query returns 0 results, auto-simplify and retry.
- Query analytics logged to help diagnose search quality over time.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from loguru import logger

from src.agents.research_ontology_agent import ResearchOntology


@dataclass
class QueryResult:
    query: str
    query_type: str     # A/B/C/D/E/F/G
    papers_found: int = 0
    after_retry: bool = False
    retry_from: Optional[str] = None


@dataclass
class QueryBatch:
    """A group of queries with analytics."""
    results: List[QueryResult] = field(default_factory=list)

    @property
    def total_queries(self) -> int:
        return len(self.results)

    @property
    def productive_queries(self) -> int:
        return sum(1 for r in self.results if r.papers_found > 0)

    @property
    def total_papers(self) -> int:
        return sum(r.papers_found for r in self.results)

    def log_analytics(self):
        logger.info(
            f"Query Analytics: {self.productive_queries}/{self.total_queries} productive queries, "
            f"{self.total_papers} total candidate papers"
        )
        for r in self.results:
            retry_note = f" [retry from '{r.retry_from}']" if r.after_retry else ""
            status = "✅" if r.papers_found > 0 else "❌"
            logger.info(
                f"  {status} [{r.query_type}] '{r.query}' → {r.papers_found} papers{retry_note}"
            )


class SearchQueryBuilder:
    """
    Converts a ResearchOntology into a prioritized list of arXiv search queries.
    Pure Python — deterministic — no LLM calls.
    """

    MAX_QUERY_WORDS = 5        # Hard cap — long queries hurt arXiv recall
    MAX_QUERIES_PER_TYPE = 4   # Per query type cap
    RETRY_MAX_DEPTH = 3        # How many words to strip in fallback

    def build_queries(self, ontology: ResearchOntology) -> List[Tuple[str, str]]:
        """
        Generate all search queries from the ontology.
        Returns List[(query_string, query_type_label)].

        Query Types:
          A — Named frameworks (exact name searches — highest precision)
          B — Core term combos (fundamental terminology)
          C — Task + core term (problem-specific)
          D — Framework + dataset (benchmark-driven)
          E — Synonyms (alternative phrasings)
          F — Methods + core (technique-specific)
          G — Broad fallback (wide recall safety net)
        """
        queries: List[Tuple[str, str]] = []

        # TYPE A: Named frameworks — search exact names (appear in paper titles)
        for fw in ontology.named_frameworks[:self.MAX_QUERIES_PER_TYPE]:
            q = self._clean(fw)
            if q:
                queries.append((q, "A:framework"))

        # TYPE B: Core terms (1-2 combined)
        for i, term in enumerate(ontology.core_terms[:self.MAX_QUERIES_PER_TYPE]):
            q = self._clip(term)
            if q:
                queries.append((q, "B:core_term"))
            # Pair first two core terms
            if i == 0 and len(ontology.core_terms) > 1:
                combined = self._clip(f"{ontology.core_terms[0]} {ontology.core_terms[1]}")
                if combined:
                    queries.append((combined, "B:core_combo"))

        # TYPE C: Task types + first core term anchor
        anchor = self._clip(ontology.core_terms[0]) if ontology.core_terms else ""
        for task in ontology.task_types[:self.MAX_QUERIES_PER_TYPE]:
            task_clean = self._clip(task)
            if task_clean and anchor:
                q = self._clip(f"{task_clean} {anchor}")
                if q:
                    queries.append((q, "C:task"))
            elif task_clean:
                queries.append((task_clean, "C:task"))

        # TYPE D: Framework + dataset (benchmark-driven searches)
        for fw in ontology.named_frameworks[:2]:
            for ds in ontology.benchmark_datasets[:2]:
                q = self._clip(f"{fw} {ds}")
                if q:
                    queries.append((q, "D:framework_dataset"))

        # TYPE E: Synonyms (alternative phrasings)
        for syn in ontology.synonyms[:self.MAX_QUERIES_PER_TYPE]:
            q = self._clip(syn)
            if q:
                queries.append((q, "E:synonym"))

        # TYPE F: Methods + anchor
        for method in ontology.methods[:self.MAX_QUERIES_PER_TYPE]:
            method_clean = self._clip(method)
            if method_clean and anchor:
                q = self._clip(f"{method_clean} {anchor}")
                if q:
                    queries.append((q, "F:method"))

        # TYPE G: Broad fallback — just the raw topic words for safety
        if ontology.core_terms:
            queries.append((self._clip(ontology.core_terms[0]), "G:broad_fallback"))

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q, qtype in queries:
            if q.lower() not in seen and len(q.strip()) > 1:
                seen.add(q.lower())
                unique_queries.append((q, qtype))

        logger.info(
            f"Query Builder produced {len(unique_queries)} unique queries "
            f"from ontology ({len(ontology.named_frameworks)} frameworks, "
            f"{len(ontology.core_terms)} core terms, "
            f"{len(ontology.task_types)} tasks)"
        )
        for q, qt in unique_queries:
            logger.debug(f"  [{qt}] '{q}'")

        return unique_queries

    def build_fallback_chain(self, query: str) -> List[str]:
        """
        Generate progressively simpler versions of a query for retry.
        Strips one word at a time from the right.

        Example:
          "episodic memory mechanisms agent" →
          ["episodic memory mechanisms", "episodic memory", "episodic"]
        """
        words = query.strip().split()
        chain = []
        for n in range(len(words) - 1, 0, -1):
            simplified = " ".join(words[:n])
            if simplified and simplified.lower() != query.lower():
                chain.append(simplified)
            if len(chain) >= self.RETRY_MAX_DEPTH:
                break
        return chain

    def _clean(self, text: str) -> str:
        """Remove markdown, punctuation noise, and strip whitespace."""
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'^\d+[\.\)]\s*', '', text)
        text = text.strip('"\'- ')
        return text.strip()

    def _clip(self, text: str) -> str:
        """Clean and enforce MAX_QUERY_WORDS word limit."""
        cleaned = self._clean(text)
        words = cleaned.split()
        return " ".join(words[:self.MAX_QUERY_WORDS])


query_builder = SearchQueryBuilder()

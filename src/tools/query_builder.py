"""
Query Builder — builds arXiv search queries from ResearchOntology.
"""

from typing import List, Tuple
from loguru import logger

from src.agents.research_ontology_agent import ResearchOntology


class QueryBuilder:
    def build_queries(self, ontology: ResearchOntology) -> List[Tuple[str, str]]:
        """
        Returns list of (query_string, query_type).
        Types: core | related | topic
        """
        queries: List[Tuple[str, str]] = []
        seen = set()

        def add(q: str, qtype: str):
            q = (q or "").strip()
            if not q:
                return
            key = q.lower()
            if key in seen:
                return
            seen.add(key)
            queries.append((q, qtype))

        # 1. Core terms (highest priority)
        for term in (ontology.core_terms or [])[:6]:
            add(term, "core")

        # 2. Related methods / techniques
        for term in (getattr(ontology, "related_terms", None) or [])[:8]:
            add(term, "related")

        # 3. Light combinations from core (optional, limited)
        cores = (ontology.core_terms or [])[:3]
        if len(cores) >= 2:
            add(f"{cores[0]} {cores[1]}", "core")

        logger.info(f"Query Builder → {len(queries)} queries")
        for q, qt in queries:
            logger.info(f"  [{qt}] {q}")

        return queries

    def build_fallback_chain(self, query: str) -> List[str]:
        """Simpler variants if a query returns 0 hits."""
        q = (query or "").strip()
        if not q:
            return []

        parts = q.split()
        fallbacks = []

        # Drop last token
        if len(parts) > 2:
            fallbacks.append(" ".join(parts[:-1]))

        # First two tokens
        if len(parts) >= 2:
            fallbacks.append(" ".join(parts[:2]))

        # First token only
        if parts:
            fallbacks.append(parts[0])

        # Dedup while preserving order
        out, seen = [], set()
        for f in fallbacks:
            if f.lower() not in seen and f.lower() != q.lower():
                seen.add(f.lower())
                out.append(f)
        return out


query_builder = QueryBuilder()
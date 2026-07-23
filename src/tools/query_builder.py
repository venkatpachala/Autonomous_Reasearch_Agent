# src/tools/query_builder.py
import re
from typing import List, Tuple
from loguru import logger
from src.agents.research_ontology_agent import ResearchOntology


class SearchQueryBuilder:
    MAX_QUERY_WORDS = 4
    MAX_QUERIES_TOTAL = 12

    def build_queries(self, ontology: ResearchOntology) -> List[Tuple[str, str]]:
        queries = []

        # Priority 1: Named frameworks (highest precision)
        for fw in ontology.named_frameworks[:5]:
            q = self._clip(fw)
            if q:
                queries.append((q, "A:framework"))

        # Priority 2: Core terms
        for term in ontology.core_terms[:6]:
            q = self._clip(term)
            if q:
                queries.append((q, "B:core"))

        # Priority 3: Methods
        for method in ontology.methods[:4]:
            q = self._clip(method)
            if q:
                queries.append((q, "C:method"))

        # Priority 4: Datasets
        for ds in ontology.benchmark_datasets[:3]:
            q = self._clip(ds)
            if q:
                queries.append((q, "D:dataset"))

        # Deduplicate
        seen = set()
        unique = []
        for q, qtype in queries:
            key = q.lower()
            if key and key not in seen:
                seen.add(key)
                unique.append((q, qtype))
            if len(unique) >= self.MAX_QUERIES_TOTAL:
                break

        logger.info(f"Query Builder → {len(unique)} queries")
        for q, qt in unique:
            logger.debug(f"  [{qt}] '{q}'")

        return unique

    def _clip(self, text: str) -> str:
        text = re.sub(r'[\*\"\'\-]+', ' ', text)
        text = re.sub(r'^\d+[\.\)]\s*', '', text)
        words = text.strip().split()
        return " ".join(words[:self.MAX_QUERY_WORDS]).strip()

    def build_fallback_chain(self, query: str) -> List[str]:
        words = query.strip().split()
        chain = []
        for n in range(len(words)-1, 0, -1):
            simplified = " ".join(words[:n])
            if simplified.lower() != query.lower():
                chain.append(simplified)
            if len(chain) >= 2:
                break
        return chain


query_builder = SearchQueryBuilder()
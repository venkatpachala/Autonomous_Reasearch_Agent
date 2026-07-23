# src/tools/query_builder.py

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from loguru import logger
from src.agents.research_ontology_agent import ResearchOntology


@dataclass
class QueryResult:
    query: str
    query_type: str
    papers_found: int = 0
    after_retry: bool = False
    retry_from: Optional[str] = None


class SearchQueryBuilder:
    MAX_QUERY_WORDS = 4
    
    def build_queries(self, ontology: ResearchOntology) -> Dict[str, List[Tuple[str, str]]]:
        """
        Produce short, high-precision arXiv keyword queries grouped by Priority Tiers.
        This enables adaptive search (stop early if P1 yields enough papers).
        
        P1 (Highest Precision): Named Frameworks, Core Terms, Datasets
        P2 (High Precision): Methods, Tasks, Key Authors
        P3 (Safety Net): Synonyms, Venues
        """
        tiers: Dict[str, List[Tuple[str, str]]] = {
            "P1": [],
            "P2": [],
            "P3": []
        }
        seen = set()

        def _add(tier: str, text: str, qtype: str):
            q = self._clip(text)
            key = q.lower().strip()
            if key and key not in seen and len(q.split()) <= self.MAX_QUERY_WORDS:
                seen.add(key)
                tiers[tier].append((q, qtype))

        # --- P1: Highest Precision ---
        for fw in ontology.named_frameworks[:4]:
            _add("P1", fw, "A:framework")
            
        for ds in ontology.benchmark_datasets[:3]:
            _add("P1", ds, "D:dataset")
            
        for term in ontology.core_terms[:3]:
            _add("P1", term, "B:core")

        # --- P2: Methods, Tasks, Authors ---
        for method in ontology.methods[:3]:
            _add("P2", method, "C:method")
            
        for task in ontology.task_types[:3]:
            _add("P2", task, "E:task")
            
        for author in ontology.key_authors[:2]:
            if ontology.core_terms:
                # E.g. "Lewis RAG"
                _add("P2", f"{author} {ontology.core_terms[0]}", "F:author")

        # --- P3: Synonyms, Venues (Safety net) ---
        for syn in ontology.synonyms[:2]:
            _add("P3", syn, "G:synonym")
            
        for venue in ontology.venues[:2]:
            if ontology.core_terms:
                _add("P3", f"{ontology.core_terms[0]} {venue}", "H:venue")

        # Log analytics
        total = sum(len(q) for q in tiers.values())
        logger.info(f"Query Builder → {total} queries across 3 tiers (P1:{len(tiers['P1'])}, P2:{len(tiers['P2'])}, P3:{len(tiers['P3'])})")
        for t, qs in tiers.items():
            for q, qt in qs:
                logger.debug(f"  [{t}] [{qt}] '{q}'")
                
        return tiers

    def build_fallback_chain(self, query: str) -> List[str]:
        """Progressively shorter versions (right-to-left strip)."""
        words = query.strip().split()
        chain = []
        for n in range(len(words) - 1, 0, -1):
            simplified = " ".join(words[:n])
            if simplified.lower() != query.lower():
                chain.append(simplified)
            if len(chain) >= 2:
                break
        return chain

    def _clean(self, text: str) -> str:
        text = re.sub(r'[\*\"\'\-]+', ' ', text)
        text = re.sub(r'^\d+[\.\)]\s*', '', text)
        return text.strip()

    def _clip(self, text: str) -> str:
        cleaned = self._clean(text)
        words = cleaned.split()
        return " ".join(words[:self.MAX_QUERY_WORDS])


query_builder = SearchQueryBuilder()
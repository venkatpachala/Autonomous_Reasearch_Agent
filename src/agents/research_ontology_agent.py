# src/agents/research_ontology_agent.py
from typing import List
from pydantic import BaseModel, Field
from loguru import logger
import json as _json
import re
import asyncio

from src.gateway import gateway


class ResearchOntology(BaseModel):
    topic_summary: str = Field(...)
    core_terms: List[str] = Field(..., min_length=3, max_length=8)
    named_frameworks: List[str] = Field(default_factory=list)
    task_types: List[str] = Field(default_factory=list)
    benchmark_datasets: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    negative_terms: List[str] = Field(default_factory=list)


class ResearchOntologyAgent:
    async def generate(self, topic: str) -> ResearchOntology:
        example = '''{
  "topic_summary": "Research on retrieval quality and evaluation for retrieval-augmented generation systems.",
  "core_terms": ["retrieval quality", "dense retrieval", "reranking", "query expansion"],
  "named_frameworks": ["RAG", "ColBERT", "DPR", "ANCE"],
  "task_types": ["passage retrieval", "document ranking", "multi-hop retrieval"],
  "benchmark_datasets": ["BEIR", "MS MARCO", "HotpotQA", "MTEB"],
  "methods": ["dense retrieval", "sparse retrieval", "hybrid retrieval", "reranking"],
  "synonyms": ["retrieval effectiveness", "information retrieval quality"],
  "negative_terms": ["memory management", "cache eviction", "RAM allocation", "operating system"]
}'''

        system = f"""You are a senior AI research librarian.

STRICT RULES (never break):
1. Output ONLY valid JSON. No markdown, no explanations.
2. NEVER invent framework names, acronyms, or systems.
   - Only include named_frameworks that actually appear in published papers.
   - If unsure, return empty list for named_frameworks.
3. core_terms must be short and common in paper titles.
4. negative_terms should include generic CS/OS terms.

Example for a retrieval topic:
{example}
"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Research Topic: {topic}\n\nOutput ONLY the JSON."}
        ]

        for attempt in range(3):
            try:
                response = await gateway.generate(
                    task="keyword_generation",
                    messages=messages,
                    temperature=0.05,
                )
                raw = response.text.strip()
                # Clean markdown
                if "```" in raw:
                    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                    if match:
                        raw = match.group(1)

                start, end = raw.find("{"), raw.rfind("}") + 1
                if start != -1 and end > start:
                    raw = raw[start:end]

                parsed = _json.loads(raw)
                ontology = ResearchOntology.model_validate(parsed)

                logger.success(f"Ontology generated for '{topic}'")
                return ontology

            except Exception as e:
                logger.warning(f"Ontology attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(1.2)

        logger.error("Ontology failed. Using minimal fallback.")
        return ResearchOntology(
            topic_summary=f"Research on {topic}",
            core_terms=[topic],
            named_frameworks=[],
            task_types=[],
            benchmark_datasets=[],
            methods=[],
            synonyms=[],
            negative_terms=["memory management", "cache eviction"]
        )


research_ontology_agent = ResearchOntologyAgent()
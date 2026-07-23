# src/agents/extractor_agent.py  (simplified version)

from typing import List, Optional
from pydantic import BaseModel, Field
from loguru import logger
from src.gateway import gateway


class EntityNode(BaseModel):
    name: str
    type: str   # Method | Dataset | Metric | Concept


class RelationshipEdge(BaseModel):
    source: str
    target: str
    relation: str
    value: Optional[str] = None


class GraphKnowledge(BaseModel):
    entities: List[EntityNode] = Field(default_factory=list)
    relationships: List[RelationshipEdge] = Field(default_factory=list)


import json
import re

class ExtractorAgent:
    def _repair_json(self, text: str) -> str:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        text = text.strip()
        text = text.replace("'", '"')
        text = re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', text) # fix unescaped backslashes
        text = re.sub(r',\s*([\]}])', r'\1', text) # trailing commas
        return text

    async def extract(self, paper_id: str, title: str, contributions: List[str], benchmarks: Optional[List[str]] = None) -> GraphKnowledge:
        """
        Lightweight extraction. Uses only title + contributions + benchmarks.
        """
        contrib_text = "\n".join(f"- {c}" for c in (contributions or [])[:6])
        bench_text = "\n".join(f"- {b}" for b in (benchmarks or [])[:6]) if benchmarks else "None"

        system_entities = """You are a knowledge graph extractor.
Return ONLY valid JSON matching this schema:
{
  "entities": [{"name": "...", "type": "Method|Dataset|Metric|Concept"}]
}
Rules:
- Output pure JSON only. No markdown, no explanation.
- Escape any backslashes properly.
- Keep names short and normalized (e.g. "RAG", "BEIR", "dense retrieval").
- Maximum 8 entities.
"""

        user_entities = f"""Title: {title}

Key Contributions:
{contrib_text}

Benchmarks:
{bench_text}

Extract entities now."""

        messages_ent = [
            {"role": "system", "content": system_entities},
            {"role": "user", "content": user_entities}
        ]

        entities = []
        try:
            resp_ent = await gateway.generate(
                task="graph_extraction",
                messages=messages_ent,
                temperature=0.0,
                retries=2
            )
            raw_ent = self._repair_json(resp_ent.text)
            parsed_ent = json.loads(raw_ent)
            entities = [EntityNode(**e) for e in parsed_ent.get("entities", [])]
        except Exception as e:
            logger.error(f"Entity extraction failed for {paper_id}: {e}")

        if not entities:
            return GraphKnowledge()

        entities_json = json.dumps([e.model_dump() for e in entities], indent=2)

        system_rels = """You are a knowledge graph extractor.
Return ONLY valid JSON matching this schema:
{
  "relationships": [{"source": "...", "target": "...", "relation": "...", "value": null}]
}
Rules:
- Output pure JSON only. No markdown, no explanation.
- Escape any backslashes properly.
- Maximum 10 relationships.
- ONLY use entities from the provided list.
"""

        user_rels = f"""Title: {title}

Key Contributions:
{contrib_text}

Benchmarks:
{bench_text}

Entities:
{entities_json}

Extract relationships now."""

        messages_rel = [
            {"role": "system", "content": system_rels},
            {"role": "user", "content": user_rels}
        ]

        relationships = []
        try:
            resp_rel = await gateway.generate(
                task="graph_extraction",
                messages=messages_rel,
                temperature=0.0,
                retries=2
            )
            raw_rel = self._repair_json(resp_rel.text)
            parsed_rel = json.loads(raw_rel)
            relationships = [RelationshipEdge(**r) for r in parsed_rel.get("relationships", [])]
        except Exception as e:
            logger.error(f"Relationship extraction failed for {paper_id}: {e}")

        logger.success(f"Graph extracted for {paper_id}: {len(entities)} entities, {len(relationships)} edges")
        return GraphKnowledge(entities=entities, relationships=relationships)


extractor_agent = ExtractorAgent()
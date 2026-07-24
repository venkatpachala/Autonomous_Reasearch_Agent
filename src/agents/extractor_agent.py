"""
Extractor Agent — Graph Entity & Relationship Extraction
Works on FULL parsed paper text. Coerces numeric relationship values to str.
"""

from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator
from loguru import logger
import json
import re

from src.gateway import gateway


class EntityNode(BaseModel):
    name: str = Field(..., description="Normalized entity name, e.g. 'RAG', 'BEIR'")
    type: str = Field(..., description="Method | Dataset | Metric | Concept")
    description: Optional[str] = Field(None, description="Short context (optional)")


class RelationshipEdge(BaseModel):
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation: str = Field(
        ..., description="Relation label, e.g. PROPOSES, EVALUATED_ON, IMPROVES"
    )
    value: Optional[str] = Field(
        None, description="Optional quantitative value as a string, e.g. '0.92' or '9.0'"
    )

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value_to_str(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        # float / int / bool → string
        return str(v)


class GraphKnowledge(BaseModel):
    entities: List[EntityNode] = Field(default_factory=list)
    relationships: List[RelationshipEdge] = Field(default_factory=list)


class ExtractorAgent:
    """Extracts entities and relationships from full paper text."""

    async def extract(
        self,
        paper_id: str,
        title: str,
        full_text: str = "",
        contributions: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None,
    ) -> GraphKnowledge:
        if not full_text and not contributions:
            logger.warning(f"No content for graph extraction on {paper_id}")
            return GraphKnowledge()

        content_parts = [f"Title: {title}"]
        if contributions:
            content_parts.append(
                "Key Contributions:\n" + "\n".join(f"- {c}" for c in contributions[:6])
            )
        if benchmarks:
            content_parts.append("Benchmarks: " + ", ".join(benchmarks[:8]))
        if full_text:
            content_parts.append(f"Paper Content (excerpt):\n{full_text[:4500]}")

        context = "\n\n".join(content_parts)

        system = """You are a knowledge graph extractor for academic papers.

Return ONLY valid JSON matching this exact schema:
{
  "entities": [
    {"name": "...", "type": "Method|Dataset|Metric|Concept", "description": "optional short context"}
  ],
  "relationships": [
    {"source": "...", "target": "...", "relation": "PROPOSES|EVALUATED_ON|IMPROVES|USES|COMPARED_TO", "value": null}
  ]
}

Rules:
- Output pure JSON only. No markdown fences. No explanations.
- Maximum 10 entities and 12 relationships.
- Normalize names (e.g. "Retrieval-Augmented Generation" → "RAG").
- Prefer concrete technical terms over generic ones.
- If you include a quantitative value, it MUST be a JSON string (e.g. "0.92"), never a number.
- Escape any backslashes properly.
"""

        user = f"{context}\n\nExtract entities and relationships now."

        try:
            response = await gateway.generate(
                task="graph_extraction",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                retries=2,
            )

            raw = response.text.strip()

            if "```" in raw:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if match:
                    raw = match.group(1)

            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

            parsed = json.loads(raw)

            # Defensive coerce before validate (in case LLM still emits numbers)
            for rel in parsed.get("relationships") or []:
                if isinstance(rel, dict) and "value" in rel and rel["value"] is not None:
                    if not isinstance(rel["value"], str):
                        rel["value"] = str(rel["value"])

            graph = GraphKnowledge.model_validate(parsed)

            logger.success(
                f"Graph extracted for {paper_id}: "
                f"{len(graph.entities)} entities, {len(graph.relationships)} edges"
            )
            return graph

        except Exception as e:
            logger.error(f"Graph extraction failed for {paper_id}: {e}")
            return GraphKnowledge()


extractor_agent = ExtractorAgent()
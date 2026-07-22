from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger
from src.gateway import gateway
from src.models.schemas import PerPaperOutput

class EntityNode(BaseModel):
    name: str = Field(..., description="Normalized entity name, e.g., 'Qwen-2.5', 'MMLU', 'Accuracy', 'LangGraph'")
    type: str = Field(..., description="Type: 'Method' (algorithm/tool/model), 'Dataset' (benchmark/test-set), 'Metric' (accuracy score/latency), 'Concept' (theory/field)")
    description: Optional[str] = Field(None, description="Short context details")

class RelationshipEdge(BaseModel):
    source: str = Field(..., description="Name of the source node")
    target: str = Field(..., description="Name of the target node")
    relation: str = Field(..., description="Relationship label, e.g. 'PROPOSES', 'EVALUATED_ON', 'IMPROVES', 'MENTIONS'")
    value: Optional[str] = Field(None, description="Quantitative score or metric value if available, e.g. '84.2%', '120ms'")

class GraphKnowledge(BaseModel):
    entities: List[EntityNode] = Field(default_factory=list)
    relationships: List[RelationshipEdge] = Field(default_factory=list)

class ExtractorAgent:
    """
    LLM-powered Entity and Relationship extraction agent.
    Parses structural paper summaries to build property graph triplets.
    """
    async def extract_graph_elements(self, output: PerPaperOutput) -> GraphKnowledge:
        paper = output.metadata
        summary = output.summary

        if not summary:
            logger.warning(f"No summary found for {output.paper_id}. Skipping entity extraction.")
            return GraphKnowledge()

        # Build comprehensive summary context
        contributions = "\n".join(f"- {c}" for c in summary.key_contributions)
        benchmarks = "\n".join(f"- {b}" for b in summary.benchmarks)
        
        system_content = """You are a specialized AI Knowledge Graph Engineer.
Your goal is to parse structured academic paper summaries and extract exact Entity-Relationship triplets.

Rules:
1. Normalize node names: use title case and remove extra spaces or punctuation (e.g. 'gpt4' ➔ 'GPT-4', 'mmlu' ➔ 'MMLU').
2. Keep entity types strictly to: 'Method', 'Dataset', 'Metric', or 'Concept'.
3. Extract quantitative relationships where possible, detailing scores/percentages in the 'value' field.
4. Output a valid JSON conforming to the requested schema. Do not generate markdown wrapping."""

        human_content = f"""Paper Title: {paper.title}
Abstract: {paper.abstract}
Objective: {summary.objective}
Methodology: {summary.methodology}
Key Contributions:
{contributions}
Benchmarks/Achievements:
{benchmarks}

Extract the graph entities and relationships."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]

        try:
            response = await gateway.generate(
                task="evaluation",  # Routes to default Ollama model locally
                messages=messages,
                temperature=0.1,
                schema_model=GraphKnowledge
            )
            if response.structured:
                logger.success(f"Extracted {len(response.structured.entities)} nodes & {len(response.structured.relationships)} edges for {output.paper_id}")
                return response.structured
        except Exception as e:
            logger.error(f"Entity extraction failed for {output.paper_id}: {e}")
            
        return GraphKnowledge()

extractor_agent = ExtractorAgent()

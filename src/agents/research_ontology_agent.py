# src/agents/research_ontology_agent.py
"""
Research Ontology Agent
Generates structured domain ontology for a research topic:
  - core_terms
  - related_terms (methods / techniques)
  - negative_terms
"""

from typing import List
from pydantic import BaseModel, Field
from loguru import logger

from src.gateway import gateway
from src.observability.tracing import traced


class ResearchOntology(BaseModel):
    """Structured ontology for guiding search + relevance filtering."""

    core_terms: List[str] = Field(
        ...,
        min_length=3,
        max_length=8,
        description="Short terms common in paper titles for this topic.",
    )
    related_terms: List[str] = Field(
        default_factory=list,
        description=(
            "Related methods, techniques, systems, and adjacent concepts "
            "(e.g. quantization, GGUF, KV cache, FlashAttention, pruning)."
        ),
    )
    negative_terms: List[str] = Field(
        default_factory=list,
        description="Generic CS/OS terms that often cause false positives.",
    )
    description: str = Field(
        default="",
        description="One-sentence scope of the research topic.",
    )


class ResearchOntologyAgent:
    @traced(name="research_ontology_agent", run_type="chain")
    async def generate(self, topic: str) -> ResearchOntology:
        messages = [
            {
                "role": "system",
                "content": (
                    "You build a research ontology for academic paper search and filtering.\n\n"
                    "Rules:\n"
                    "1. core_terms: 3–8 short phrases that appear in real paper titles.\n"
                    "2. related_terms: 6–15 related methods, algorithms, systems, formats, "
                    "and techniques researchers would consider on-topic even if the exact "
                    "topic phrase is missing.\n"
                    "   Examples for 'running local models': quantization, QAT, GPTQ, AWQ, "
                    "GGUF, llama.cpp, vLLM, KV cache, FlashAttention, pruning, low VRAM, "
                    "on-device inference, edge LLM.\n"
                    "3. negative_terms: generic OS/infra terms that hijack search "
                    "(e.g. RAM allocation, process scheduling) when they are NOT the topic.\n"
                    "4. Keep terms short and search-friendly.\n"
                    "5. Do not invent unrelated domains.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    "Generate core_terms, related_terms, and negative_terms for this topic."
                ),
            },
        ]

        try:
            response = await gateway.generate(
                task="ontology",
                messages=messages,
                temperature=0.2,
                schema_model=ResearchOntology,
            )
            if response.structured:
                ont = response.structured
                logger.info(
                    f"Ontology for '{topic}': "
                    f"{len(ont.core_terms)} core, "
                    f"{len(ont.related_terms)} related, "
                    f"{len(ont.negative_terms)} negative"
                )
                return ont
        except Exception as e:
            logger.warning(f"Ontology generation failed: {e}. Using fallback.")

        # Fallback — minimal but usable
        return ResearchOntology(
            core_terms=[topic],
            related_terms=[],
            negative_terms=["memory management", "cache eviction"],
            description=f"Fallback ontology for {topic}",
        )


research_ontology_agent = ResearchOntologyAgent()
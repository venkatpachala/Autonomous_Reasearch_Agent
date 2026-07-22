"""
Synthesis Agent: Cross-paper reasoning, trend analysis, and research insight generation.
Activated for collection-level queries instead of single-document RAG.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from loguru import logger

from src.gateway import gateway
from src.models.schemas import KnowledgeNote


class PaperRelationship(BaseModel):
    paper_a: str = Field(..., description="Short title or arXiv ID of paper A")
    paper_b: str = Field(..., description="Short title or arXiv ID of paper B")
    relationship: Literal["extends", "contradicts", "evaluates_same_data", "proposes_alternative", "builds_upon", "surveys"]
    description: str = Field(..., description="One sentence explaining the relationship.")


class ResearchSynthesis(BaseModel):
    """Structured cross-paper analysis output."""
    research_directions: List[str] = Field(
        ...,
        description="2-5 distinct research directions or sub-themes identified across the papers."
    )
    key_methods: List[str] = Field(
        ...,
        description="The most important methods, architectures, or algorithms discussed."
    )
    common_datasets_benchmarks: List[str] = Field(
        default_factory=list,
        description="Datasets or benchmarks that appear across multiple papers."
    )
    emerging_trends: List[str] = Field(
        ...,
        description="2-4 trends that are gaining momentum based on the papers."
    )
    research_gaps: List[str] = Field(
        ...,
        description="2-4 open problems or things not yet addressed by any paper in the collection."
    )
    paper_relationships: List[PaperRelationship] = Field(
        default_factory=list,
        description="Notable relationships between papers (extensions, contradictions, evaluations)."
    )
    state_of_the_art: str = Field(
        ...,
        description="Which paper/approach represents the current state of the art and why."
    )
    synthesis_narrative: str = Field(
        ...,
        description=(
            "A rich, 4-6 paragraph research synthesis narrative that groups papers into themes, "
            "explains how they relate, identifies the progression of ideas, notes contradictions, "
            "and highlights what is missing. Write like an expert researcher, NOT like a list. "
            "Include specific citations using [arXiv:ID - Short Title] format."
        )
    )


class SynthesisAgent:
    """
    Generates cross-paper research synthesis, trend analysis, and gap identification.
    Used for collection-level queries like 'what are all these papers about?'
    """

    def _build_collection_context(self, notes: List[KnowledgeNote]) -> str:
        """Format all notes into a compact collection context string."""
        parts = []
        for i, note in enumerate(notes, 1):
            sd = note.structured_data
            contributions = (
                "\n".join(f"    - {c}" for c in sd.key_contributions[:3])
                if sd and sd.key_contributions else "    - N/A"
            )
            limitations = (
                ", ".join(sd.limitations[:2])
                if sd and sd.limitations else "Not specified"
            )
            benchmarks = (
                str(sd.benchmarks[:2])
                if sd and sd.benchmarks else "Not specified"
            )
            parts.append(
                f"Paper {i}: [{note.paper_id}] {note.title}\n"
                f"  Summary: {note.one_sentence_summary}\n"
                f"  Objective: {sd.objective if sd else 'N/A'}\n"
                f"  Key Contributions:\n{contributions}\n"
                f"  Benchmarks: {benchmarks}\n"
                f"  Limitations: {limitations}\n"
                f"  Concepts: {', '.join(note.concepts[:6])}\n"
            )
        return "\n\n".join(parts)

    async def synthesize(
        self,
        notes: List[KnowledgeNote],
        query: str,
        topic: str,
        graph_triplets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a cross-paper research synthesis.
        Returns a structured dict with the synthesis and metadata.
        """
        if not notes:
            return {
                "answer": "No papers are indexed for this topic yet. Ingest papers first.",
                "sources": [],
                "synthesis": None
            }

        collection_context = self._build_collection_context(notes)
        n_papers = len(notes)

        graph_section = ""
        if graph_triplets:
            graph_section = (
                f"\nKnowledge Graph Relationships:\n"
                + "\n".join(f"  - {t}" for t in graph_triplets[:20])
                + "\n"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a world-class AI research analyst with deep expertise in synthesizing academic literature.\n\n"
                    "You are given a collection of research papers and must produce an expert-level research synthesis.\n\n"
                    "Your synthesis should:\n"
                    "1. Group papers into coherent research directions/themes\n"
                    "2. Identify how papers build on, extend, or contradict each other\n"
                    "3. Extract emerging trends and methodological progressions\n"
                    "4. Identify clear research gaps and open problems\n"
                    "5. Cite specific papers using [arXiv:ID - Short Title] format\n"
                    "6. Write as an expert researcher would — thematic, analytical, NOT a list of summaries\n\n"
                    "Do NOT simply summarize each paper individually. BUILD CONNECTIONS."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Research Topic: {topic}\n"
                    f"User Query: {query}\n"
                    f"Number of Papers in Collection: {n_papers}\n"
                    f"{graph_section}\n"
                    f"Paper Collection:\n\n{collection_context}\n\n"
                    "Generate a comprehensive research synthesis."
                )
            }
        ]

        try:
            response = await gateway.generate(
                task="synthesis",
                messages=messages,
                temperature=0.3,
                schema_model=ResearchSynthesis
            )

            if response.structured:
                synthesis = response.structured

                # Build the final human-readable answer
                answer = self._format_synthesis_answer(synthesis, query)

                sources = [
                    {"paper_id": note.paper_id, "title": note.title,
                     "arxiv_url": f"https://arxiv.org/abs/{note.paper_id}",
                     "score": note.criticality_score}
                    for note in notes
                ]

                logger.success(
                    f"Synthesis complete: {len(synthesis.research_directions)} directions, "
                    f"{len(synthesis.research_gaps)} gaps, {len(synthesis.paper_relationships)} relationships"
                )

                return {
                    "answer": answer,
                    "sources": sources,
                    "synthesis": synthesis.model_dump(),
                    "contexts_used": n_papers
                }

        except Exception as e:
            logger.error(f"Synthesis generation failed: {e}")

        # Fallback: plain narrative synthesis
        fallback_answer = await self._fallback_narrative(collection_context, query, topic)
        return {
            "answer": fallback_answer,
            "sources": [{"paper_id": n.paper_id, "title": n.title} for n in notes],
            "synthesis": None,
            "contexts_used": n_papers
        }

    def _format_synthesis_answer(self, synthesis: ResearchSynthesis, query: str) -> str:
        """Format structured synthesis into a rich markdown answer."""
        lines = []

        lines.append("## Research Synthesis\n")
        lines.append(synthesis.synthesis_narrative)

        lines.append("\n---\n")
        lines.append("### Research Directions")
        for i, d in enumerate(synthesis.research_directions, 1):
            lines.append(f"{i}. {d}")

        lines.append("\n### Emerging Trends")
        for t in synthesis.emerging_trends:
            lines.append(f"- {t}")

        lines.append("\n### Research Gaps & Open Problems")
        for g in synthesis.research_gaps:
            lines.append(f"- {g}")

        if synthesis.state_of_the_art:
            lines.append(f"\n### State of the Art\n{synthesis.state_of_the_art}")

        if synthesis.paper_relationships:
            lines.append("\n### Key Paper Relationships")
            for rel in synthesis.paper_relationships[:5]:
                lines.append(
                    f"- **{rel.paper_a}** `{rel.relationship.upper()}` **{rel.paper_b}**: {rel.description}"
                )

        return "\n".join(lines)

    async def _fallback_narrative(self, context: str, query: str, topic: str) -> str:
        """Simple narrative fallback when structured output fails."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert research analyst. Synthesize the provided papers into "
                    "a coherent narrative that groups themes, identifies trends, and finds gaps. "
                    "Write 3-4 analytical paragraphs. Do NOT just summarize each paper separately."
                )
            },
            {
                "role": "user",
                "content": f"Topic: {topic}\nQuery: {query}\n\nPapers:\n{context}"
            }
        ]
        try:
            response = await gateway.generate(
                task="synthesis", messages=messages, temperature=0.3
            )
            return response.text
        except Exception as e:
            return f"Synthesis failed: {e}"


synthesis_agent = SynthesisAgent()

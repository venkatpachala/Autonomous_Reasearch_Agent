"""
Synthesis Agent: Cross-paper reasoning, trend analysis, and research insight generation.
Updated to work with chunk-based dicts (new architecture) while remaining compatible
with legacy KnowledgeNote objects.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from loguru import logger

from src.gateway import gateway


class PaperRelationship(BaseModel):
    paper_a: str = Field(..., description="Short title or arXiv ID of paper A")
    paper_b: str = Field(..., description="Short title or arXiv ID of paper B")
    relationship: Literal[
        "extends", "contradicts", "evaluates_same_data",
        "proposes_alternative", "builds_upon", "surveys"
    ]
    description: str = Field(..., description="One sentence explaining the relationship.")
    evidence_basis: str = Field(
        default="inferred",
        description="Exact quote or section that supports this relationship, or 'inferred'."
    )


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
        description="Notable relationships between papers."
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
            "and highlights what is missing. Write like an expert researcher. "
            "Include specific citations using [arXiv:ID - Short Title] format."
        )
    )


class SynthesisAgent:
    """
    Generates cross-paper research synthesis, trend analysis, and gap identification.
    Compatible with both:
      - List[dict]          (current chunk-based storage)
      - List[KnowledgeNote] (legacy path)
    """

    def _get_field(self, note: Any, field: str, default=None):
        """Safe accessor for both dict and object notes."""
        if isinstance(note, dict):
            return note.get(field, default)
        return getattr(note, field, default)

    def _build_sources(self, notes: List[Any]) -> List[Dict]:
        """Build sources list that works for both dicts and KnowledgeNote objects."""
        sources = []
        for n in notes:
            paper_id = self._get_field(n, "paper_id", "unknown")
            title = self._get_field(n, "title", "Untitled")
            score = self._get_field(n, "score") or self._get_field(n, "criticality_score")

            sources.append({
                "paper_id": paper_id,
                "title": title,
                "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
                "score": score,
            })
        return sources

    def _build_collection_context(self, notes: List[Any]) -> str:
        """
        Format all notes into a compact collection context string.
        Handles both dict chunks and legacy KnowledgeNote objects.
        """
        parts = []
        for i, note in enumerate(notes, 1):
            paper_id = self._get_field(note, "paper_id", "unknown")
            title = self._get_field(note, "title", "Untitled")
            is_dict = isinstance(note, dict)

            sd = None if is_dict else getattr(note, "structured_data", None)

            if sd:
                # Rich path — real KnowledgeNote with structured_data
                contributions = (
                    "\n".join(f"    - {c}" for c in sd.key_contributions[:3])
                    if sd.key_contributions else "    - N/A"
                )
                limitations = ", ".join(sd.limitations[:2]) if sd.limitations else "Not specified"
                benchmarks = str(sd.benchmarks[:2]) if sd.benchmarks else "Not specified"
                summary = getattr(note, "one_sentence_summary", "N/A")
                concepts = ", ".join(getattr(note, "concepts", [])[:6]) or "N/A"

                parts.append(
                    f"Paper {i}: [{paper_id}] {title}\n"
                    f"  Summary: {summary}\n"
                    f"  Objective: {sd.objective}\n"
                    f"  Key Contributions:\n{contributions}\n"
                    f"  Benchmarks: {benchmarks}\n"
                    f"  Limitations: {limitations}\n"
                    f"  Concepts: {concepts}\n"
                )
            else:
                # Common path today — raw chunk content
                content = self._get_field(note, "content") or str(note)
                num_chunks = self._get_field(note, "num_chunks")
                chunk_note = f" ({num_chunks} chunks)" if num_chunks else ""

                parts.append(
                    f"Paper {i}: [{paper_id}] {title}{chunk_note}\n"
                    f"  Content excerpt:\n{content[:2500]}\n"
                )
        return "\n\n".join(parts)

    async def synthesize(
        self,
        notes: List[Any],
        query: str,
        topic: str,
        graph_triplets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a cross-paper research synthesis.
        Accepts List[dict] or List[KnowledgeNote].
        """
        if not notes:
            return {
                "answer": "No papers are indexed for this topic yet. Ingest papers first.",
                "sources": [],
                "synthesis": None,
                "contexts_used": 0
            }

        collection_context = self._build_collection_context(notes)
        n_papers = len(notes)
        sources = self._build_sources(notes)

        graph_section = ""
        if graph_triplets:
            graph_section = (
                "\nKnowledge Graph Relationships:\n"
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
                    "CRITICAL RULES FOR RELATIONSHIPS:\n"
                    "- Only claim a relationship if there is explicit textual evidence in the provided paper content.\n"
                    "- If you cannot find a direct connection, set evidence_basis to 'inferred'.\n"
                    "- Do NOT fabricate citation connections between papers that never reference each other.\n\n"
                    "Do NOT simply summarize each paper individually. BUILD CONNECTIONS based on the evidence."
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
                answer = self._format_synthesis_answer(synthesis, query)

                logger.success(
                    f"Synthesis complete: {len(synthesis.research_directions)} directions, "
                    f"{len(synthesis.research_gaps)} gaps, "
                    f"{len(synthesis.paper_relationships)} relationships"
                )

                return {
                    "answer": answer,
                    "sources": sources,
                    "synthesis": synthesis.model_dump(),
                    "contexts_used": n_papers
                }

        except Exception as e:
            logger.error(f"Structured synthesis failed: {e}")

        # Fallback: plain narrative synthesis
        fallback_answer = await self._fallback_narrative(collection_context, query, topic)
        return {
            "answer": fallback_answer,
            "sources": sources,
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
                    "Write 3-4 analytical paragraphs. Do NOT just summarize each paper separately. "
                    "Cite papers as [arXiv:ID - Short Title]."
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
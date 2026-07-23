"""
Critic + Knowledge Note Generator v2
======================================
Generates a rich KnowledgeNote from the paper's extracted content and summary.
Extracts: detailed_summary, concepts, hardware_devices, table_summaries,
key_quotes, criticality_score.

v2: Real LLM call via gateway (previously a placeholder stub).
"""

import json
import re
from loguru import logger

from src.gateway import gateway
from src.models.schemas import KnowledgeNote, PerPaperOutput, PaperStatus, StructuredPaperSummary
from src.observability.tracing import traced


class CriticNoteAgent:
    def __init__(self):
        pass  # Uses centralized gateway

    @traced(name="critic_note_agent", run_type="chain")
    async def run(self, output: PerPaperOutput) -> PerPaperOutput:
        paper = output.metadata
        summary = output.summary

        # Build content for the LLM — PDF text is richer than abstract
        content = ""
        if output.extracted and output.extracted.text:
            content = output.extracted.text[:5000].strip()
        if not content and paper.abstract:
            content = paper.abstract

        # Pull known benchmarks/hardware from summarizer output (if available)
        known_benchmarks = summary.benchmarks if summary else []
        known_contributions = summary.key_contributions if summary else []
        known_hardware = []
        if summary and summary.achievements and "Hardware/Devices mentioned:" in summary.achievements:
            hw_line = summary.achievements.split("Hardware/Devices mentioned:")[-1].strip()
            known_hardware = [h.strip() for h in hw_line.split(",") if h.strip()]

        system_prompt = (
            "You are a research librarian creating a rich knowledge note for long-term retrieval.\n\n"
            "Return ONLY valid JSON matching exactly this schema:\n"
            "{\n"
            '  "one_sentence_summary": "One crisp sentence capturing the paper\'s main contribution.",\n'
            '  "detailed_summary": "3-5 sentence rich markdown summary covering objective, method, results.",\n'
            '  "concepts": ["concept1", "concept2", "concept3"],\n'
            '  "hardware_devices": ["device1", "device2"],\n'
            '  "table_summaries": ["Table 1: description", "Table 2: description"],\n'
            '  "key_quotes": ["notable quote or claim from paper"],\n'
            '  "criticality_score": 0.75,\n'
            '  "tags": ["tag1", "tag2"]\n'
            "}\n\n"
            "Rules:\n"
            "- concepts: specific technical entities (methods, architectures, datasets, metrics). Max 10.\n"
            "- hardware_devices: name every specific device, chip, GPU, NPU, SoC, or board mentioned. Empty if none.\n"
            "- table_summaries: briefly describe each table's purpose if visible in the content.\n"
            "- criticality_score: 0.0-1.0 float rating novelty/impact (0.9=breakthrough, 0.5=incremental, 0.2=minor).\n"
            "- Output ONLY raw JSON. No markdown, no explanation."
        )

        user_prompt = (
            f"Paper Title: {paper.title}\n"
            f"Abstract: {paper.abstract[:500] if paper.abstract else 'N/A'}\n\n"
            f"Known Contributions: {'; '.join(known_contributions[:3])}\n"
            f"Known Benchmarks: {', '.join(known_benchmarks[:5])}\n"
            f"Known Hardware: {', '.join(known_hardware[:5]) if known_hardware else 'None identified yet'}\n\n"
            f"Paper Content:\n{content[:3000]}\n\n"
            "Generate the knowledge note now."
        )

        try:
            response = await gateway.generate(
                task="summarization",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                retries=2
            )
            raw = response.text.strip()

            if "```" in raw:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if match:
                    raw = match.group(1)

            start, end = raw.find("{"), raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

            parsed = json.loads(raw)

            # Build the structured_data — merge with existing summary or create a minimal one
            structured = summary or StructuredPaperSummary(
                objective=paper.abstract[:300] if paper.abstract else "Not specified.",
                methodology="Not extracted.",
                key_contributions=known_contributions or ["Not extracted."],
                achievements="Not extracted.",
                benchmarks=known_benchmarks,
                limitations=[],
                future_work=[]
            )

            output.knowledge_note = KnowledgeNote(
                paper_id=output.paper_id,
                title=paper.title,
                one_sentence_summary=parsed.get("one_sentence_summary", paper.title),
                detailed_summary=parsed.get("detailed_summary", paper.abstract[:600] if paper.abstract else ""),
                structured_data=structured,
                concepts=parsed.get("concepts", [])[:12],
                hardware_devices=parsed.get("hardware_devices", known_hardware)[:8],
                table_summaries=parsed.get("table_summaries", [])[:6],
                key_quotes=parsed.get("key_quotes", [])[:3],
                criticality_score=float(parsed.get("criticality_score", 0.65)),
                tags=parsed.get("tags", ["research", "ai"])[:6],
            )

            logger.success(
                f"Knowledge Note created for {output.paper_id}: "
                f"{len(output.knowledge_note.concepts)} concepts, "
                f"{len(output.knowledge_note.hardware_devices)} hardware, "
                f"{len(output.knowledge_note.table_summaries)} tables"
            )

        except Exception as e:
            logger.warning(f"LLM critic failed for {output.paper_id}: {e}. Using stub.")
            output.knowledge_note = self._stub_note(output, summary, known_hardware, known_benchmarks)

        output.status = PaperStatus.COMPLETED
        return output

    def _stub_note(self, output, summary, hardware, benchmarks) -> KnowledgeNote:
        paper = output.metadata
        structured = summary or StructuredPaperSummary(
            objective=paper.abstract[:300] if paper.abstract else "Not extracted.",
            methodology="Not extracted.",
            key_contributions=["Not extracted.", "Not extracted.", "Not extracted."],
            achievements="Not extracted.",
            benchmarks=benchmarks,
            limitations=[],
            future_work=[]
        )
        return KnowledgeNote(
            paper_id=output.paper_id,
            title=paper.title,
            one_sentence_summary=paper.abstract[:250] if paper.abstract else paper.title,
            detailed_summary=f"{paper.title}. {paper.abstract[:600] if paper.abstract else ''}",
            structured_data=structured,
            concepts=[],
            hardware_devices=hardware,
            table_summaries=[],
            key_quotes=[],
            criticality_score=0.65,
            tags=["research", "ai"],
        )


critic_agent = CriticNoteAgent()

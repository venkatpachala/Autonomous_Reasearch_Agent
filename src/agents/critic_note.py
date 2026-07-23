"""
Critic + Knowledge Note Generator v2
Updated to safely access full extracted content.
"""

import json
import re
from loguru import logger

from src.gateway import gateway
from src.models.schemas import KnowledgeNote, PerPaperOutput, PaperStatus, StructuredPaperSummary
from src.observability.tracing import traced


class CriticNoteAgent:
    def __init__(self):
        pass

    @traced(name="critic_note_agent", run_type="chain")
    async def run(self, output: PerPaperOutput) -> PerPaperOutput:
        paper = output.metadata
        summary = output.summary

        # SAFE FULL TEXT EXTRACTION
        extracted_text = ""
        if output.extracted:
            extracted_text = (
                getattr(output.extracted, "full_text", "")
                or getattr(output.extracted, "text", "")
                or getattr(output.extracted, "text_content", "")
                or getattr(output.extracted, "content", "")
                or ""
            )

        # Pull known benchmarks/hardware from summarizer
        known_benchmarks = summary.benchmarks if summary else []
        known_contributions = summary.key_contributions if summary else []
        known_hardware = []

        system_prompt = (
            "You are a research librarian creating a rich knowledge note for long-term retrieval.\n\n"
            "Return ONLY valid JSON matching exactly this schema:\n"
            "{\n"
            '  "one_sentence_summary": "...",\n'
            '  "detailed_summary": "...",\n'
            '  "concepts": ["concept1", "concept2"],\n'
            '  "hardware_devices": ["device1", "device2"],\n'
            '  "table_summaries": ["Table 1: description"],\n'
            '  "key_quotes": ["quote"],\n'
            '  "criticality_score": 0.75,\n'
            '  "tags": ["tag1", "tag2"]\n'
            "}\n\n"
            "Rules:\n"
            "- Output ONLY raw JSON. No markdown.\n"
            "- concepts: specific technical entities.\n"
            "- hardware_devices: name every device/chip mentioned.\n"
            "- criticality_score: 0.0-1.0 float."
        )

        user_prompt = (
            f"Paper Title: {paper.title}\n"
            f"Abstract: {paper.abstract[:500] if paper.abstract else 'N/A'}\n\n"
            f"Full Content (first 4000 chars):\n{extracted_text[:4000]}\n\n"
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
                detailed_summary=parsed.get("detailed_summary", extracted_text[:600]),
                structured_data=structured,
                concepts=parsed.get("concepts", [])[:12],
                hardware_devices=parsed.get("hardware_devices", known_hardware)[:8],
                table_summaries=parsed.get("table_summaries", [])[:6],
                key_quotes=parsed.get("key_quotes", [])[:3],
                criticality_score=float(parsed.get("criticality_score", 0.65)),
                tags=parsed.get("tags", ["research", "ai"])[:6],
            )

            logger.success(f"Knowledge Note created for {output.paper_id}")

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
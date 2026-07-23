"""
Structured Summarizer Agent v2
================================
Calls the LLM with actual extracted paper content to produce structured summaries.
Fields extracted: objective, methodology, key_contributions, achievements,
benchmarks, hardware_devices, limitations, future_work.

v2: Real LLM call via gateway (previously a placeholder stub).
"""

import json
import re
from loguru import logger

from src.gateway import gateway
from src.models.schemas import StructuredPaperSummary, PerPaperOutput, PaperStatus


class SummarizerAgent:
    def __init__(self):
        pass  # Uses centralized gateway for LLM routing

    async def run(self, output: PerPaperOutput) -> PerPaperOutput:
        paper = output.metadata

        # Build the content to summarize (PDF text > abstract fallback)
        content = ""
        if output.extracted and output.extracted.text:
            content = output.extracted.text[:4000].strip()

        if not content and paper.abstract:
            content = paper.abstract

        if not content:
            output.summary = self._stub_summary(paper)
            output.status = PaperStatus.SUMMARIZING
            return output

        system_prompt = (
            "You are a precise research paper analyst.\n"
            "Extract a structured summary from the given paper content.\n\n"
            "Return ONLY valid JSON matching exactly this schema:\n"
            "{\n"
            '  "objective": "One precise sentence describing the main research goal.",\n'
            '  "methodology": "2-3 sentences describing the approach/method used.",\n'
            '  "key_contributions": ["contribution 1", "contribution 2", "contribution 3"],\n'
            '  "achievements": "Main results and what was achieved (include numbers if mentioned).",\n'
            '  "benchmarks": ["benchmark/dataset name 1", "benchmark/dataset name 2"],\n'
            '  "hardware_devices": ["device or chip name 1", "device or chip name 2"],\n'
            '  "limitations": ["limitation 1", "limitation 2"],\n'
            '  "future_work": ["future direction 1"]\n'
            "}\n\n"
            "Rules:\n"
            "- Output ONLY raw JSON. No markdown, no explanation.\n"
            "- hardware_devices: list specific hardware (GPUs, NPUs, chips, edge devices, SoCs) mentioned.\n"
            "- benchmarks: list dataset/benchmark names (e.g. MMLU, WikiText-2, ImageNet, BEIR).\n"
            "- key_contributions: minimum 3 distinct technical contributions. Be specific.\n"
            "- If information for a field is not present in the text, use an empty list [] or 'Not specified.'"
        )

        user_prompt = (
            f"Paper Title: {paper.title}\n\n"
            f"Paper Content:\n{content}\n\n"
            "Extract the structured summary now."
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

            # Strip markdown if present
            if "```" in raw:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if match:
                    raw = match.group(1)

            start, end = raw.find("{"), raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

            parsed = json.loads(raw)
            hardware = parsed.get("hardware_devices", [])
            achievements_text = parsed.get("achievements", "Not specified.")
            if hardware:
                achievements_text += f"\n\nHardware/Devices mentioned: {', '.join(hardware)}"

            output.summary = StructuredPaperSummary(
                objective=parsed.get("objective", "Not specified."),
                methodology=parsed.get("methodology", "Not specified."),
                key_contributions=parsed.get("key_contributions", [])[:5],
                achievements=achievements_text,
                benchmarks=parsed.get("benchmarks", [])[:8],
                limitations=parsed.get("limitations", [])[:4],
                future_work=parsed.get("future_work", [])[:3],
            )

            logger.success(
                f"Summarized '{paper.arxiv_id}': "
                f"{len(output.summary.key_contributions)} contributions, "
                f"{len(output.summary.benchmarks)} benchmarks, "
                f"{len(hardware)} hardware mentions"
            )

        except Exception as e:
            logger.warning(f"LLM summarization failed for {paper.arxiv_id}: {e}. Using stub.")
            output.summary = self._stub_summary(paper)

        output.status = PaperStatus.SUMMARIZING
        return output

    def _stub_summary(self, paper) -> StructuredPaperSummary:
        """Minimal fallback using abstract text only."""
        return StructuredPaperSummary(
            objective=paper.abstract[:300] if paper.abstract else "Research objective not extracted.",
            methodology="Methodology details not extracted from PDF.",
            key_contributions=[
                "Primary technical contribution (extraction failed)",
                "Novel approach over previous methods (extraction failed)",
                "Empirical results (extraction failed)"
            ],
            achievements="Results not extracted.",
            benchmarks=[],
            limitations=["Limitations not extracted."],
            future_work=["Future directions not extracted."]
        )


summarizer_agent = SummarizerAgent()
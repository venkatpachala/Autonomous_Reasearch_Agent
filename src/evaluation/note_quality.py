"""
Note Quality Evaluation using LangSmith + LLM-as-Judge
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from langsmith.evaluation import EvaluationResult

from src.config import settings
from src.models.schemas import KnowledgeNote


from pydantic import BaseModel, Field
from src.gateway import gateway
from src.models.schemas import KnowledgeNote


@dataclass
class NoteQualityScore:
    completeness: float          # 0-1
    contribution_density: float  # 0-1
    specificity: float           # 0-1
    usefulness: float            # 1-5 (LLM judge)
    overall: float               # weighted
    feedback: str = ""


class JudgeEvaluation(BaseModel):
    score: float = Field(..., ge=1.0, le=5.0, description="Evaluation score from 1 to 5")
    feedback: str = Field(..., description="Short feedback explaining the score")


@traceable(name="score_knowledge_note", run_type="chain")
def score_knowledge_note_rule_based(note: KnowledgeNote) -> Dict[str, float]:
    """Fast rule-based scoring (no LLM)."""
    # Completeness
    required_fields = [
        note.one_sentence_summary,
        note.detailed_summary,
        note.structured_data,
    ]
    completeness = sum(1 for f in required_fields if f) / len(required_fields)

    # Contribution density
    contributions = []
    if note.structured_data and note.structured_data.key_contributions:
        contributions = note.structured_data.key_contributions
    contrib_score = min(len(contributions) / 4.0, 1.0)  # target ~4 contributions

    # Specificity (very rough heuristic)
    text = (note.detailed_summary or "") + " " + (note.one_sentence_summary or "")
    has_numbers = any(c.isdigit() for c in text)
    has_methods = any(w in text.lower() for w in ["method", "architecture", "algorithm", "benchmark", "dataset", "model"])
    specificity = 0.4 + (0.3 if has_numbers else 0) + (0.3 if has_methods else 0)

    return {
        "completeness": round(completeness, 3),
        "contribution_density": round(contrib_score, 3),
        "specificity": round(specificity, 3),
    }


@traceable(name="llm_judge_note_quality", run_type="llm")
async def llm_judge_note_quality(note: KnowledgeNote) -> Dict[str, Any]:
    """LLM-as-Judge for usefulness of a KnowledgeNote using AI Gateway and structured output validation."""
    system_content = """You are a senior AI research engineer evaluating the quality of a paper summary note.
Score the note from 1.0 to 5.0 on how useful it would be for quickly understanding and using the paper later.

Criteria:
- Clarity and conciseness
- Capture of key technical contributions
- Presence of concrete details (methods, results, limitations)
- Overall value as a long-term memory item

Return a JSON object matching this schema:
{"score": <1.0-5.0>, "feedback": "<short feedback>"}"""

    contributions = []
    if note.structured_data and note.structured_data.key_contributions:
        contributions = note.structured_data.key_contributions

    human_content = f"""Title: {note.title}

One-sentence: {note.one_sentence_summary}

Detailed Summary:
{(note.detailed_summary or "")[:2000]}

Key Contributions: {contributions}

Evaluate this KnowledgeNote."""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": human_content}
    ]

    try:
        response = await gateway.generate(
            task="evaluation",
            messages=messages,
            temperature=0.1,
            schema_model=JudgeEvaluation
        )
        if response.structured:
            return {"score": response.structured.score, "feedback": response.structured.feedback}
        
        # Fallback manual parsing if schema wasn't fully enforced
        content = response.text
        score = 3.0
        feedback = content
        if "score" in content.lower():
            import re
            match = re.search(r'"score"\s*:\s*([\d\.]+)', content)
            if match:
                score = float(match.group(1))
        return {"score": score, "feedback": feedback}
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return {"score": 3.0, "feedback": "Judge failed"}


@traceable(name="full_note_quality", run_type="chain")
async def evaluate_note_quality(note: KnowledgeNote) -> NoteQualityScore:
    """Full note quality evaluation (rules + LLM judge)."""
    rules = score_knowledge_note_rule_based(note)
    judge = await llm_judge_note_quality(note)

    usefulness = judge["score"]  # 1-5
    # Normalize usefulness to 0-1 for overall
    usefulness_norm = (usefulness - 1) / 4.0

    overall = (
        0.25 * rules["completeness"] +
        0.25 * rules["contribution_density"] +
        0.20 * rules["specificity"] +
        0.30 * usefulness_norm
    )

    return NoteQualityScore(
        completeness=rules["completeness"],
        contribution_density=rules["contribution_density"],
        specificity=rules["specificity"],
        usefulness=usefulness,
        overall=round(overall, 3),
        feedback=judge.get("feedback", "")
    )


"""
LangSmith-compatible Evaluators
"""

from typing import Dict, Any, Optional
from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith.schemas import Example, Run
from loguru import logger

from src.evaluation.note_quality import evaluate_note_quality, score_knowledge_note_rule_based
from src.models.schemas import KnowledgeNote


def note_completeness_evaluator(run: Run, example: Optional[Example] = None) -> EvaluationResult:
    """Simple rule-based completeness evaluator for LangSmith."""
    try:
        # Expect the run outputs to contain a knowledge_note dict
        outputs = run.outputs or {}
        note_data = outputs.get("knowledge_note") or outputs.get("note")
        if not note_data:
            return EvaluationResult(key="note_completeness", score=0.0, comment="No note found")

        note = KnowledgeNote(**note_data) if isinstance(note_data, dict) else note_data
        scores = score_knowledge_note_rule_based(note)
        return EvaluationResult(
            key="note_completeness",
            score=scores["completeness"],
            comment=f"Contributions: {scores['contribution_density']:.2f}, Specificity: {scores['specificity']:.2f}"
        )
    except Exception as e:
        logger.warning(f"note_completeness_evaluator failed: {e}")
        return EvaluationResult(key="note_completeness", score=0.0, comment=str(e))


def note_overall_quality_evaluator(run: Run, example: Optional[Example] = None) -> EvaluationResult:
    """Placeholder for full note quality (async version needs special handling)."""
    try:
        outputs = run.outputs or {}
        note_data = outputs.get("knowledge_note") or outputs.get("note")
        if not note_data:
            return EvaluationResult(key="note_quality", score=0.0)

        note = KnowledgeNote(**note_data) if isinstance(note_data, dict) else note_data
        scores = score_knowledge_note_rule_based(note)
        # Simple weighted overall without LLM for now (fast)
        overall = (
            0.4 * scores["completeness"] +
            0.3 * scores["contribution_density"] +
            0.3 * scores["specificity"]
        )
        return EvaluationResult(
            key="note_quality",
            score=overall,
            comment=f"completeness={scores['completeness']}, density={scores['contribution_density']}"
        )
    except Exception as e:
        return EvaluationResult(key="note_quality", score=0.0, comment=str(e))


# Faithfulness evaluator (for final answers)
def faithfulness_evaluator(run: Run, example: Optional[Example] = None) -> EvaluationResult:
    """
    Very lightweight faithfulness check.
    In production we would use a proper LLM-as-judge or RAGAS.
    """
    try:
        outputs = run.outputs or {}
        answer = outputs.get("answer", "")
        sources = outputs.get("sources", [])

        if not answer:
            return EvaluationResult(key="faithfulness", score=0.0, comment="No answer")

        # Simple heuristic: if we have sources and the answer is long enough, assume decent faithfulness
        has_sources = len(sources) > 0
        has_citations = "arXiv" in answer or "arxiv" in answer.lower() or "[" in answer

        score = 0.5
        if has_sources:
            score += 0.25
        if has_citations:
            score += 0.25

        return EvaluationResult(
            key="faithfulness",
            score=min(score, 1.0),
            comment=f"sources={len(sources)}, has_citations={has_citations}"
        )
    except Exception as e:
        return EvaluationResult(key="faithfulness", score=0.0, comment=str(e))


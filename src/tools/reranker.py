"""
Cross-encoder reranker for RAG candidate chunks.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        name = "BAAI/bge-reranker-base"
        logger.info(f"Loading reranker: {name}")
        _model = CrossEncoder(name)
        logger.success(f"Reranker ready: {name}")
    return _model


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 8,
    text_key: str = "content",
) -> List[Dict[str, Any]]:
    """
    Rerank candidate chunks by (query, text) relevance.
    Each candidate is a dict with at least `text_key`.
    Adds/updates `score` with reranker score.
    """
    if not candidates:
        return []
    if not (query or "").strip():
        return candidates[:top_k]

    if len(candidates) <= top_k:
        # Still score so scores are comparable, optional; or just return
        pass

    model = _get_model()
    pairs = []
    for c in candidates:
        text = (c.get(text_key) or c.get("text") or "")[:4000]
        pairs.append([query, text])

    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.warning(f"Reranker failed ({e}); returning original order")
        return candidates[:top_k]

    scored = []
    for c, s in zip(candidates, scores):
        item = dict(c)
        item["score"] = float(s)
        item["rerank_score"] = float(s)
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
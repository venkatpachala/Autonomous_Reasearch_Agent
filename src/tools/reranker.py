import time
from loguru import logger

_model = None

def _get_model():
    global _model
    if _model is None:
        t = time.perf_counter()
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("BAAI/bge-reranker-base")  # your model id
        logger.info(f"Reranker model load ms={(time.perf_counter()-t)*1000:.0f}")
    return _model

def rerank(query: str, candidates: list, top_k: int = 8, text_key: str = "content"):
    t0 = time.perf_counter()

    t = time.perf_counter()
    model = _get_model()
    load_or_reuse_ms = (time.perf_counter() - t) * 1000

    pairs = [(query, (c.get(text_key) or c.get("content") or "")[:2000]) for c in candidates]

    t = time.perf_counter()
    scores = model.predict(pairs, batch_size=16)  # batch, not one-by-one
    infer_ms = (time.perf_counter() - t) * 1000

    for c, s in zip(candidates, scores):
        c = c  # mutate copy if needed
        c["score"] = float(s)

    ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    total = (time.perf_counter() - t0) * 1000
    logger.info(
        f"Rerank detail ms | get_model={load_or_reuse_ms:.0f} "
        f"infer={infer_ms:.0f} n={len(pairs)} total={total:.0f}"
    )
    return ranked
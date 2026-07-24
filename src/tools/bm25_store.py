"""
Lightweight BM25 index over chunk texts, persisted per topic.
Updated when papers are stored; used for hybrid retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.config import settings

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    logger.warning("rank_bm25 not installed. Run: pip install rank-bm25")

_TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", re.I)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if len(t) > 1]


def _topic_slug(topic: str) -> str:
    return (topic or "global").lower().strip().replace(" ", "_").replace("/", "_")


class BM25Store:
    """
    One BM25 corpus per topic under {base}/bm25/{topic_slug}.json
    Records: chunk_id, text, metadata
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base = Path(base_dir or getattr(settings, "base_dir", Path("."))) / "bm25"
        self.base.mkdir(parents=True, exist_ok=True)
        # cache: topic_slug -> {records, bm25, tokens}
        self._cache: Dict[str, Any] = {}

    def _path(self, topic: str) -> Path:
        return self.base / f"{_topic_slug(topic)}.json"

    def _load(self, topic: str) -> List[Dict[str, Any]]:
        path = self._path(topic)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"BM25 load failed for {topic}: {e}")
            return []

    def _save(self, topic: str, records: List[Dict[str, Any]]):
        path = self._path(topic)
        path.write_text(
            json.dumps(records, ensure_ascii=False),
            encoding="utf-8",
        )

    def _rebuild(self, topic: str, records: List[Dict[str, Any]]):
        if BM25Okapi is None or not records:
            self._cache[topic] = {"records": records, "bm25": None}
            return
        corpus = [_tokenize(r.get("text") or "") for r in records]
        # Avoid empty docs
        corpus = [c if c else ["empty"] for c in corpus]
        bm25 = BM25Okapi(corpus)
        self._cache[_topic_slug(topic)] = {
            "records": records,
            "bm25": bm25,
        }

    def _ensure(self, topic: str):
        key = _topic_slug(topic)
        if key not in self._cache:
            records = self._load(topic)
            self._rebuild(topic, records)

    def add_chunks(
        self,
        topic: str,
        chunks: List[Dict[str, Any]],
    ):
        """
        chunks: [{chunk_id, text, metadata}, ...]
        Upserts by chunk_id for this topic.
        """
        if not chunks:
            return
        key = _topic_slug(topic)
        records = self._load(topic)
        by_id = {r["chunk_id"]: r for r in records}

        for c in chunks:
            cid = c.get("chunk_id") or c.get("id")
            if not cid:
                continue
            text = c.get("text") or ""
            meta = dict(c.get("metadata") or {})
            by_id[cid] = {
                "chunk_id": cid,
                "text": text,
                "metadata": meta,
            }

        records = list(by_id.values())
        self._save(topic, records)
        self._rebuild(topic, records)
        logger.debug(f"BM25 index '{key}': {len(records)} chunks")

    def search(
        self,
        query: str,
        topic: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        if BM25Okapi is None or not (query or "").strip():
            return []

        topic = topic or "global"
        self._ensure(topic)
        key = _topic_slug(topic)
        entry = self._cache.get(key) or {}
        records = entry.get("records") or []
        bm25 = entry.get("bm25")
        if not records or bm25 is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        out = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            r = records[i]
            meta = r.get("metadata") or {}
            out.append({
                "paper_id": meta.get("paper_id"),
                "title": meta.get("title", "Untitled"),
                "content": r.get("text") or "",
                "score": float(scores[i]),
                "bm25_score": float(scores[i]),
                "chunk_id": r.get("chunk_id"),
                "arxiv_url": f"https://arxiv.org/abs/{meta.get('paper_id')}",
                "chunk_type": meta.get("chunk_type"),
                "section": meta.get("section"),
                "source": "bm25",
            })
        return out


bm25_store = BM25Store()
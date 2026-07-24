"""
Research Index - Paper registry for dedup, monitoring, and metadata lookup.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from loguru import logger

from src.config import settings


class ResearchIndex:
    """
    JSON-backed index of processed papers and topics.

    papers[arxiv_id] = {
      paper_id, title, authors, abstract, published, categories,
      topics, pdf_path, status, first_seen, last_processed
    }
    topics[topic_key] = {
      paper_ids, last_monitored, last_ingestion
    }
    """

    def __init__(self, index_path: Path = None):
        self.index_path = index_path or (settings.base_dir / "research_index.json")
        self.data = self._load()

    def _load(self) -> Dict:
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                data.setdefault("papers", {})
                data.setdefault("topics", {})
                return data
            except Exception as e:
                logger.warning(f"Failed to load research index: {e}")
        return {"papers": {}, "topics": {}}

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(self.data, indent=2, default=str),
            encoding="utf-8",
        )

    def _save(self):
        """Alias used by newer call sites."""
        self.save()

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #
    def is_paper_known(self, arxiv_id: str) -> bool:
        return arxiv_id in self.data.get("papers", {})

    def get_known_paper_ids(self) -> Set[str]:
        return set(self.data.get("papers", {}).keys())

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Return full metadata for one paper, or None."""
        if not paper_id:
            return None
        return self.data.get("papers", {}).get(paper_id)

    def get_topic_papers(self, topic: str) -> List[str]:
        topic_key = topic.lower().strip()
        return list(
            self.data.get("topics", {}).get(topic_key, {}).get("paper_ids", [])
        )

    def get_papers_for_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Return metadata dicts for all papers under a topic."""
        ids = self.get_topic_papers(topic)
        papers = self.data.get("papers", {})
        return [papers[pid] for pid in ids if pid in papers]

    def get_last_monitored(self, topic: str) -> Optional[datetime]:
        topic_key = topic.lower().strip()
        ts = self.data.get("topics", {}).get(topic_key, {}).get("last_monitored")
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                return None
        return None

    def get_all_topics(self) -> List[str]:
        return list(self.data.get("topics", {}).keys())

    def list_titles(self, topic: Optional[str] = None) -> List[Dict[str, str]]:
        if topic:
            papers = self.get_papers_for_topic(topic)
        else:
            papers = list(self.data.get("papers", {}).values())
        return [
            {
                "paper_id": p.get("paper_id") or p.get("arxiv_id", ""),
                "title": p.get("title", "Untitled"),
            }
            for p in papers
        ]

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register_paper(
        self,
        arxiv_id: str,
        title: str,
        topic: str,
        authors: Optional[List[str]] = None,
        abstract: Optional[str] = None,
        published: Optional[str] = None,
        categories: Optional[List[str]] = None,
        pdf_path: Optional[str] = None,
        status: str = "indexed",):
        """Register or update a paper with full metadata."""
        now = datetime.utcnow().isoformat()
        topic_key = (topic or "unknown").lower().strip()
        papers = self.data.setdefault("papers", {})
        existing = papers.get(arxiv_id, {})

        #       Merge topics list
        topics_list = list(existing.get("topics") or [])
        if topic_key and topic_key not in topics_list:
            topics_list.append(topic_key)

        papers[arxiv_id] = {
            "paper_id": arxiv_id,
            "title": title or existing.get("title") or arxiv_id,
            "authors": (
                authors if authors is not None else existing.get("authors", [])
            ),
            "abstract": (
                abstract if abstract is not None else existing.get("abstract", "")
            ),
            "published": (
                published if published is not None else existing.get("published", "")
            ),
            "categories": (
                categories
                if categories is not None
                else existing.get("categories", [])
            ),
            "topics": topics_list,
            "topic": topic_key,
            "pdf_path": pdf_path or existing.get("pdf_path"),
            "status": status or existing.get("status", "indexed"),
            "first_seen": existing.get("first_seen") or now,
            "last_processed": now,
            "updated_at": now,
            # Stage 5: preserve graph enrichment state across re-ingest
            "graph_status": existing.get("graph_status"),
            "graph_error": existing.get("graph_error"),
        }

        # Topic index
        topics = self.data.setdefault("topics", {})
        if topic_key not in topics:
            topics[topic_key] = {
                "paper_ids": [],
                "last_monitored": None,
                "last_ingestion": now,
            }
        entry = topics[topic_key]
        entry.setdefault("paper_ids", [])
        if arxiv_id not in entry["paper_ids"]:
            entry["paper_ids"].append(arxiv_id)
        entry["last_ingestion"] = now

        self.save()
        logger.debug(f"Registered paper metadata: {arxiv_id}")
    
    def mark_topic_monitored(self, topic: str):
        topic_key = topic.lower().strip()
        topics = self.data.setdefault("topics", {})
        if topic_key not in topics:
            topics[topic_key] = {
                "paper_ids": [],
                "last_monitored": None,
                "last_ingestion": None,
            }
        topics[topic_key]["last_monitored"] = datetime.utcnow().isoformat()
        self.save()

    def stats(self) -> Dict:
        return {
            "total_papers": len(self.data.get("papers", {})),
            "total_topics": len(self.data.get("topics", {})),
            "topics": {
                t: len(info.get("paper_ids", []))
                for t, info in self.data.get("topics", {}).items()
            },
        }
    
    def set_graph_status(self, paper_id: str, status: str, error: str = None):
        p = self.data.get("papers", {}).get(paper_id)
        if not p:
            return
        p["graph_status"] = status
        if error is not None:
            p["graph_error"] = error
        elif "graph_error" in p and status == "completed":
            p.pop("graph_error", None)
        self.save()


research_index = ResearchIndex()
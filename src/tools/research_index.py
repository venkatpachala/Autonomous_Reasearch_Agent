"""
Research Index - Tracks which papers have been processed and for which topics.
Prevents re-ingestion and enables continuous monitoring.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime
from loguru import logger

from src.config import settings


class ResearchIndex:
    """
    Simple but effective index stored as JSON.
    
    Structure:
    {
      "papers": {
        "2506.06962v3": {
          "title": "...",
          "topics": ["agentic RAG memory systems"],
          "first_seen": "...",
          "last_processed": "..."
        }
      },
      "topics": {
        "agentic RAG memory systems": {
          "paper_ids": ["2506.06962v3", ...],
          "last_monitored": "2026-07-10T...",
          "last_ingestion": "..."
        }
      }
    }
    """

    def __init__(self, index_path: Path = None):
        self.index_path = index_path or (settings.base_dir / "research_index.json")
        self.data = self._load()

    def _load(self) -> Dict:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load research index: {e}")
        return {"papers": {}, "topics": {}}

    def save(self):
        self.index_path.write_text(
            json.dumps(self.data, indent=2, default=str),
            encoding="utf-8"
        )

    def is_paper_known(self, arxiv_id: str) -> bool:
        return arxiv_id in self.data["papers"]

    def get_known_paper_ids(self) -> Set[str]:
        return set(self.data["papers"].keys())

    def get_topic_papers(self, topic: str) -> List[str]:
        topic_key = topic.lower().strip()
        return self.data["topics"].get(topic_key, {}).get("paper_ids", [])

    def get_last_monitored(self, topic: str) -> Optional[datetime]:
        topic_key = topic.lower().strip()
        ts = self.data["topics"].get(topic_key, {}).get("last_monitored")
        if ts:
            return datetime.fromisoformat(ts)
        return None

    def register_paper(self, arxiv_id: str, title: str, topic: str):
        now = datetime.utcnow().isoformat()
        topic_key = topic.lower().strip()

        # Paper entry
        if arxiv_id not in self.data["papers"]:
            self.data["papers"][arxiv_id] = {
                "title": title,
                "topics": [topic_key],
                "first_seen": now,
                "last_processed": now
            }
        else:
            paper = self.data["papers"][arxiv_id]
            if topic_key not in paper["topics"]:
                paper["topics"].append(topic_key)
            paper["last_processed"] = now

        # Topic entry
        if topic_key not in self.data["topics"]:
            self.data["topics"][topic_key] = {
                "paper_ids": [],
                "last_monitored": None,
                "last_ingestion": now
            }

        topic_entry = self.data["topics"][topic_key]
        if arxiv_id not in topic_entry["paper_ids"]:
            topic_entry["paper_ids"].append(arxiv_id)
        topic_entry["last_ingestion"] = now

        self.save()

    def mark_topic_monitored(self, topic: str):
        topic_key = topic.lower().strip()
        if topic_key not in self.data["topics"]:
            self.data["topics"][topic_key] = {
                "paper_ids": [],
                "last_monitored": None,
                "last_ingestion": None
            }
        self.data["topics"][topic_key]["last_monitored"] = datetime.utcnow().isoformat()
        self.save()

    def get_all_topics(self) -> List[str]:
        return list(self.data["topics"].keys())

    def stats(self) -> Dict:
        return {
            "total_papers": len(self.data["papers"]),
            "total_topics": len(self.data["topics"]),
            "topics": {
                t: len(info.get("paper_ids", []))
                for t, info in self.data["topics"].items()
            }
        }


research_index = ResearchIndex()


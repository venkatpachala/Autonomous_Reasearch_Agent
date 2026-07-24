"""
Session Manager - Handles Research Sessions (topic-scoped workspaces)
Updated with paper number mapping for "paper 1 / paper 3" resolution.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger

from src.models.session import ResearchSession, ChatMessage
from src.config import settings
from src.graphs.ingestion_graph import ingestion_graph
from src.models.schemas import ResearchState


class SessionManager:
    def __init__(self, sessions_dir: Path = None):
        self.sessions_dir = sessions_dir or (settings.base_dir / "sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[ResearchSession] = None

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def list_sessions(self) -> List[ResearchSession]:
        sessions = []
        for f in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sessions.append(ResearchSession(**data))
            except Exception as e:
                logger.warning(f"Failed to load session {f.name}: {e}")
        sessions.sort(key=lambda s: s.last_active, reverse=True)
        return sessions

    def create_session(self, topic: str, description: str = None) -> ResearchSession:
        session = ResearchSession(topic=topic, description=description)
        self.save_session(session)
        self.current_session = session
        logger.success(f"Created new session {session.session_id} for topic: {topic}")
        return session

    def load_session(self, session_id: str) -> Optional[ResearchSession]:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        session = ResearchSession(**data)
        self.current_session = session
        return session

    def save_session(self, session: ResearchSession):
        path = self._session_path(session.session_id)
        path.write_text(session.model_dump_json(indent=2), encoding="utf-8")

    def get_or_create_session(self, topic: str) -> ResearchSession:
        for s in self.list_sessions():
            if s.topic.lower() == topic.lower() and s.status == "active":
                logger.info(f"Reusing existing session {s.session_id} for topic '{topic}'")
                self.current_session = s
                return s
        return self.create_session(topic)

    def build_paper_number_map(self, session: Optional[ResearchSession] = None) -> Dict[int, str]:
        """
        Stable mapping:
            1 → first paper_id in session.papers_ingested
            2 → second paper_id
            ...
        """
        s = session or self.current_session
        if not s or not s.papers_ingested:
            return {}
        return {i + 1: pid for i, pid in enumerate(s.papers_ingested)}

    async def ensure_papers_ingested(self, session: ResearchSession, force: bool = False) -> ResearchSession:
        if session.papers_ingested and not force:
            logger.info(f"Session already has {len(session.papers_ingested)} papers. Skipping ingestion.")
            return session

        logger.info(f"Running ingestion for topic: {session.topic}")

        initial_state: ResearchState = {
            "topic": session.topic,
            "keywords": [],
            "papers": [],
            "processed_papers": [],
            "messages": [],
            "status": "running",
            "current_stage": "decompose",
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            result = await ingestion_graph.ainvoke(initial_state)
            processed = result.get("processed_papers", [])
            paper_ids = []
            for p in processed:
                if isinstance(p, dict):
                    paper_ids.append(p.get("paper_id") or p.get("arxiv_id"))
                else:
                    paper_ids.append(getattr(p, "paper_id", None))

            session.papers_ingested = [pid for pid in paper_ids if pid]
            self.save_session(session)
            logger.success(f"Ingested {len(session.papers_ingested)} papers into session {session.session_id}")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise

        return session

    def add_message(self, role: str, content: str, sources: Optional[List[Dict]] = None):
        if not self.current_session:
            raise RuntimeError("No active session")
        if role == "user":
            self.current_session.add_user_message(content)
        else:
            self.current_session.add_assistant_message(content, sources)
        self.save_session(self.current_session)


session_manager = SessionManager()
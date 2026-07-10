"""
Research Session models
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: Optional[List[Dict[str, Any]]] = None


class ResearchSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    papers_ingested: List[str] = Field(default_factory=list)  # arxiv_ids
    conversation: List[ChatMessage] = Field(default_factory=list)
    status: str = "active"  # active | archived
    description: Optional[str] = None

    def add_user_message(self, content: str):
        self.conversation.append(ChatMessage(role="user", content=content))
        self.last_active = datetime.utcnow()

    def add_assistant_message(self, content: str, sources: Optional[List[Dict]] = None):
        self.conversation.append(
            ChatMessage(role="assistant", content=content, sources=sources)
        )
        self.last_active = datetime.utcnow()
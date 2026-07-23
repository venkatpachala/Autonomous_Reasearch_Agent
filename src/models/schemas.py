"""
Core Pydantic schemas and LangGraph state definition for the Research Agent.
Updated for full parsed content storage.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
import operator
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, HttpUrl


# =============================================================================
# ENUMS
# =============================================================================
class PaperStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    SUMMARIZING = "summarizing"      # optional / chat-time
    CRITIQUING = "critiquing"        # optional
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(str, Enum):
    DECOMPOSE = "decompose"
    RETRIEVE = "retrieve"
    PARALLEL_PAPER_PIPELINE = "parallel_paper_pipeline"
    MEMORY_UPDATE = "memory_update"
    QUERY = "query"
    MONITOR = "monitor"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None


class PaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    authors: List[Author]
    abstract: str
    published_date: datetime
    updated_date: Optional[datetime] = None
    pdf_url: HttpUrl
    arxiv_url: HttpUrl
    categories: List[str] = Field(default_factory=list)
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    primary_category: Optional[str] = None


class ExtractedContent(BaseModel):
    """Full parsed content from PDF — this is the source of truth"""
    full_text: str = Field(..., min_length=100)
    markdown: str = Field(default="", description="Markdown version if available")
    sections: Dict[str, str] = Field(default_factory=dict)   # heading -> content
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    page_count: int = 0


class StructuredPaperSummary(BaseModel):
    """Optional — generated on demand for chat"""
    objective: str
    methodology: str
    key_contributions: List[str] = Field(..., min_items=2)
    achievements: str
    benchmarks: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    future_work: List[str] = Field(default_factory=list)


class KnowledgeNote(BaseModel):
    """Optional rich note — not used for primary indexing"""
    paper_id: str
    title: str
    one_sentence_summary: str
    detailed_summary: str
    structured_data: Optional[StructuredPaperSummary] = None
    criticality_score: float = Field(ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# LANGGRAPH STATE
# =============================================================================
class ResearchState(TypedDict):
    topic: str
    keywords: List[str]
    research_ontology: Dict[str, Any]
    search_strategy: Optional[str]

    papers: Annotated[List[PaperMetadata], operator.add]
    papers_to_process: List[PaperMetadata]

    processed_papers: Annotated[List[Dict[str, Any]], operator.add]
    failed_papers: Annotated[List[Dict[str, Any]], operator.add]

    # New: full parsed content per paper
    extracted_contents: Annotated[List[Dict[str, Any]], operator.add]

    messages: Annotated[list, add_messages]
    current_stage: ProcessingStage
    status: Literal["running", "completed", "failed", "partial"]
    error: Optional[str]
    timestamp: datetime


# =============================================================================
# INPUT / OUTPUT FOR PER-PAPER PIPELINE
# =============================================================================
class PerPaperInput(BaseModel):
    paper: PaperMetadata
    topic: str


class PerPaperOutput(BaseModel):
    paper_id: str
    metadata: PaperMetadata
    extracted: ExtractedContent                     # ← Full parsed content (primary)
    summary: Optional[StructuredPaperSummary] = None   # ← Optional
    knowledge_note: Optional[KnowledgeNote] = None     # ← Optional
    local_pdf_path: Optional[str] = None
    status: PaperStatus = PaperStatus.EXTRACTING
    error: Optional[str] = None
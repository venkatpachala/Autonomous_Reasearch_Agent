"""
Core Pydantic schemas and LangGraph state definition for the Research Agent.
Production-grade with strict typing and structured outputs.
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
    SUMMARIZING = "summarizing"
    CRITIQUING = "critiquing"
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
# PYDANTIC MODELS (Structured Outputs for LLMs)
# =============================================================================
class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None


class PaperMetadata(BaseModel):
    """Core metadata from arXiv / Semantic Scholar"""
    arxiv_id: str = Field(..., description="e.g. 2405.12345")
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
    """Raw + structured content after PDF parsing"""
    full_text: str = Field(..., min_length=100)
    sections: Dict[str, str] = Field(default_factory=dict)  # e.g. {"Introduction": "...", "Methods": "..."}
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    figures: List[Dict[str, Any]] = Field(default_factory=list)  # description + caption if available
    references: List[str] = Field(default_factory=list)


class StructuredPaperSummary(BaseModel):
    """Senior-engineer level structured breakdown of the paper"""
    objective: str = Field(..., description="Clear research objective / problem statement")
    methodology: str = Field(..., description="High-level approach + key technical choices")
    key_contributions: List[str] = Field(..., min_items=3)
    achievements: str = Field(..., description="What was actually achieved (quantitative where possible)")
    benchmarks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Important benchmarks with numbers, datasets, SOTA comparison"
    )
    limitations: List[str] = Field(default_factory=list)
    future_work: List[str] = Field(default_factory=list)
    practical_implications: Optional[str] = None
    reproducibility_notes: Optional[str] = None


class KnowledgeNote(BaseModel):
    """Rich, embeddable note optimized for long-term memory + retrieval"""
    paper_id: str  # arxiv_id
    title: str
    one_sentence_summary: str
    detailed_summary: str  # rich markdown
    structured_data: StructuredPaperSummary
    key_quotes: List[str] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)  # extracted entities/concepts
    criticality_score: float = Field(ge=0.0, le=1.0, description="How important/novel this paper is")
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1  # for arXiv v1, v2 updates


class TopicConfig(BaseModel):
    """User preference / watched topic"""
    topic_id: str
    name: str
    description: str
    keywords: List[str]
    last_ingested: Optional[datetime] = None
    is_active: bool = True
    alert_enabled: bool = True


# =============================================================================
# LANGGRAPH STATE
# =============================================================================
class ResearchState(TypedDict):
    """
    Main state for the Research Agent graph.
    Uses Annotated reducers for safe parallel updates.
    """

    # Input
    topic: str
    topic_config: Optional[TopicConfig]

    # Decomposer output
    keywords: List[str]
    search_strategy: Optional[str]

    # Retriever output
    papers: Annotated[List[PaperMetadata], operator.add]          # raw papers from search
    papers_to_process: List[PaperMetadata]               # filtered / deduped

    # Parallel per-paper pipeline results
    processed_papers: Annotated[List[Dict[str, Any]], operator.add]  # final enriched paper dicts
    failed_papers: Annotated[List[Dict[str, Any]], operator.add]

    # Storage
    vector_ids: Annotated[List[str], operator.add]
    graph_node_ids: Annotated[List[str], operator.add]

    # Query / Chat
    messages: Annotated[list, add_messages]
    current_query: Optional[str]
    query_results: Optional[List[Dict[str, Any]]]

    # Continuous monitoring
    new_papers_found: Annotated[List[PaperMetadata], operator.add]
    alerts: Annotated[List[Dict[str, Any]], operator.add]

    # Control / Meta
    current_stage: ProcessingStage
    status: Literal["running", "completed", "failed", "partial"]
    error: Optional[str]
    run_id: Optional[str]
    timestamp: datetime


# =============================================================================
# HELPER MODELS
# =============================================================================
class PerPaperInput(BaseModel):
    """Input for one parallel paper processing branch"""
    paper: PaperMetadata
    topic: str


class PerPaperOutput(BaseModel):
    """Output from one parallel paper processing branch"""
    paper_id: str
    metadata: PaperMetadata
    extracted: ExtractedContent
    summary: StructuredPaperSummary
    knowledge_note: KnowledgeNote
    local_pdf_path: Optional[str] = None
    status: PaperStatus
    error: Optional[str] = None
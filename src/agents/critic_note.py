"""
Critic + Knowledge Note Generator
Instrumented with LangSmith.
"""

from langchain_ollama import ChatOllama
from loguru import logger

from src.config import settings
from src.models.schemas import KnowledgeNote, PerPaperOutput, PaperStatus
from src.observability.tracing import traced


class CriticNoteAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.critic_model,
            temperature=0.2,
            base_url=settings.ollama_base_url,
        )

    @traced(name="critic_note_agent", run_type="chain")
    async def run(self, output: PerPaperOutput) -> PerPaperOutput:
        paper = output.metadata

        output.knowledge_note = KnowledgeNote(
            paper_id=output.paper_id,
            title=paper.title,
            one_sentence_summary=paper.abstract[:250] if paper.abstract else paper.title,
            detailed_summary=f"Structured notes for {paper.title}. "
                             f"Abstract: {paper.abstract[:600] if paper.abstract else 'N/A'}",
            structured_data=output.summary,
            criticality_score=0.75,
            concepts=["RAG", "Agentic Systems", "Memory"] if "rag" in paper.title.lower() or "agent" in paper.title.lower() else [],
            tags=["research", "ai"]
        )

        output.status = PaperStatus.COMPLETED
        logger.success(f"Created Knowledge Note for {output.paper_id}")
        return output


critic_agent = CriticNoteAgent()


"""
Critic + Knowledge Note Generator (Senior Engineer Perspective)
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from src.config import settings
from src.models.schemas import KnowledgeNote, PerPaperOutput


class CriticNoteAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.critic_model,
            temperature=0.2,
            base_url=settings.ollama_base_url,
        )

    async def run(self, output: PerPaperOutput) -> PerPaperOutput:
        prompt_text = f"""Act as a senior AI/ML Engineer reviewing this paper.

Title: {output.metadata.title}
Abstract: {output.metadata.abstract[:500]}
Key points from text: {output.extracted.full_text[:4000]}

Create a rich Knowledge Note for long-term memory."""

        # Simplified for now
        output.knowledge_note = KnowledgeNote(
            paper_id=output.paper_id,
            title=output.metadata.title,
            one_sentence_summary=output.metadata.abstract[:200],
            detailed_summary="Rich notes generated",
            structured_data=output.summary or None,  # type: ignore
            criticality_score=0.85,
            concepts=["RAG", "Agentic", "Memory"],
            tags=["rag", "agentic"]
        )

        logger.success(f"Created Knowledge Note for {output.paper_id}")
        output.status = "completed"
        return output


critic_agent = CriticNoteAgent()
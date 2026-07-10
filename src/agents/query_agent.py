"""
Query / Chat Agent - Talks to the Research Memory using RAG.
"""

from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.config import settings
from src.tools.retriever import research_retriever


class QueryAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.default_model,
            temperature=0.2,
            base_url=settings.ollama_base_url,
        )
        self.retriever = research_retriever

    async def answer(self, question: str, topic: Optional[str] = None) -> Dict[str, Any]:
        contexts = self.retriever.search(question, topic=topic, n_results=6)

        if not contexts:
            return {
                "answer": "I don't have any relevant research papers in my knowledge base for this question yet. "
                          "Ingest more papers on this topic first.",
                "sources": [],
                "contexts_used": 0
            }

        context_str = self._format_contexts(contexts)

        system_prompt = """You are a senior AI Research Engineer with access to a personal research knowledge base of academic papers.

Answer the user's question **strictly based on the provided paper notes**.

Rules:
- Ground every claim in the provided context.
- Always cite papers using the format: [arXiv:XXXX.XXXXX - Short Title]
- If context is insufficient, say so honestly.
- Be technical, precise and insightful.
- Compare papers when multiple are relevant.
- Structure long answers clearly."""

        human_prompt = f"""Question: {question}

Relevant Research Context:
{context_str}

Answer the question using only the above context. Include citations."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            answer = response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Error generating answer: {e}"

        sources = [
            {
                "paper_id": c["paper_id"],
                "title": c["title"],
                "arxiv_url": c["arxiv_url"],
                "score": c.get("score"),
            }
            for c in contexts
        ]

        return {
            "answer": answer,
            "sources": sources,
            "contexts_used": len(contexts),
        }

    def _format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        parts = []
        for i, ctx in enumerate(contexts, 1):
            note = ctx.get("full_note")
            title = ctx["title"]
            paper_id = ctx["paper_id"]
            content = ctx["content"]

            extra = ""
            if note:
                extra = f"\nOne-sentence: {note.one_sentence_summary}"
                if note.structured_data:
                    extra += f"\nKey Contributions: {note.structured_data.key_contributions}"

            parts.append(
                f"--- Paper {i}: [{paper_id}] {title} ---\n"
                f"{content}{extra}\n"
            )
        return "\n".join(parts)


query_agent = QueryAgent()
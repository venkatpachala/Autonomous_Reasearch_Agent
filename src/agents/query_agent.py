"""
Query / Chat Agent - Talks to Helix Research using RAG.
Instrumented with LangSmith tracing.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from src.gateway import gateway
from src.tools.retriever import research_retriever
from src.observability.tracing import traced


class QueryAgent:
    """RAG-based agent that answers questions over the personal research knowledge base."""

    def __init__(self):
        self.retriever = research_retriever

    @traced(name="query_agent_answer", run_type="chain")
    async def answer(self, question: str, topic: Optional[str] = None, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """
        Main entry point: retrieve relevant papers and generate a grounded answer with citations.
        """
        # 1. Retrieve relevant context
        contexts = self.retriever.search(question, topic=topic, n_results=6)

        if not contexts:
            return {
                "answer": "I don't have any relevant research papers in my knowledge base for this question yet. "
                          "Ingest more papers on this topic first.",
                "sources": [],
                "contexts_used": 0
            }

        # 2. Build context string
        context_str = self._format_contexts(contexts)

        # 3. Build messages
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt}
        ]

        # 4. Generate answer
        try:
            response = await gateway.generate(
                task="research_answer",
                messages=messages,
                temperature=0.2
            )
            answer = response.text
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Error generating answer: {e}"

        # 5. Prepare sources
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


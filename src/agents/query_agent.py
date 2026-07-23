"""
Query / Chat Agent - Talks to Helix Research using intent-routed RAG.
Updated for chunk-based retrieval (no longer depends on KnowledgeNote objects).
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from src.gateway import gateway
from src.tools.retriever import research_retriever
from src.agents.intent_classifier import intent_classifier, QueryIntent
from src.agents.synthesis_agent import synthesis_agent
from src.db.neo4j_client import neo4j_client
from src.observability.tracing import traced


class QueryAgent:
    """Intent-routed RAG agent over the personal research knowledge base."""

    def __init__(self):
        self.retriever = research_retriever

    @traced(name="query_agent_answer", run_type="chain")
    async def answer(
        self,
        question: str,
        topic: Optional[str] = None,
        chat_history: Optional[List] = None
    ) -> Dict[str, Any]:
        intent: QueryIntent = await intent_classifier.classify(question, topic=topic)

        if intent_classifier.is_collection_level(intent):
            return await self._handle_collection_query(question, intent, topic)
        elif intent.intent == "comparison":
            return await self._handle_comparison_query(question, intent, topic)
        elif intent.intent == "expand_collection":
            return self._handle_expand_collection(question, intent, topic)
        else:
            return await self._handle_targeted_query(question, intent, topic)

    # ------------------------------------------------------------------ #
    # EXPAND COLLECTION HANDLER
    # ------------------------------------------------------------------ #
    def _handle_expand_collection(
        self, question: str, intent: QueryIntent, topic: Optional[str]
    ) -> Dict[str, Any]:
        logger.info(f"Expand collection intent detected for topic '{topic}'")
        return {
            "answer": (
                f"To fetch more papers for this session, use the `/ingest` command.\n\n"
                f"This will run the full arXiv search pipeline for the current topic "
                f"**'{topic}'** and add new papers to your knowledge base."
            ),
            "sources": [],
            "contexts_used": 0,
            "intent": "expand_collection",
            "retrieval_confidence": 1.0
        }

    # ------------------------------------------------------------------ #
    # COLLECTION-LEVEL HANDLER (overview / trends / gaps)
    # ------------------------------------------------------------------ #
    async def _handle_collection_query(
        self, question: str, intent: QueryIntent, topic: Optional[str]
    ) -> Dict[str, Any]:
        """Loads ALL chunks for the topic and runs the SynthesisAgent."""
        logger.info(f"Collection-level query ({intent.intent}) — loading all chunks for '{topic}'")

        notes = await self.retriever.get_all_notes_for_topic(topic) if topic else []

        if not notes:
            return {
                "answer": (
                    f"No papers are indexed for the topic '{topic}' yet. "
                    "Please ingest papers on this topic first using `/ingest`."
                ),
                "sources": [],
                "contexts_used": 0,
                "intent": intent.intent,
                "retrieval_confidence": 0.0
            }

        # Safe graph enrichment (works with dict-based chunks)
        graph_triplets = []
        if neo4j_client.is_connected() and notes:
            paper_ids = list({n.get("paper_id") for n in notes if n.get("paper_id")})
            graph_triplets = neo4j_client.get_related_triplets(paper_ids[:20])

        result = await synthesis_agent.synthesize(
            notes=notes,
            query=question,
            topic=topic or "research collection",
            graph_triplets=graph_triplets
        )
        result["intent"] = intent.intent
        result["retrieval_confidence"] = 1.0
        return result

    # ------------------------------------------------------------------ #
    # COMPARISON HANDLER
    # ------------------------------------------------------------------ #
    async def _handle_comparison_query(
        self, question: str, intent: QueryIntent, topic: Optional[str]
    ) -> Dict[str, Any]:
        logger.info(f"Comparison query — expanded: {intent.expanded_query[:80]}")

        retrieved = await self.retriever.search(
            intent.expanded_query, topic=topic, n_results=8
        )
        papers = retrieved.get("papers", [])
        graph_triplets = retrieved.get("graph_triplets", [])
        confidence = retrieved.get("retrieval_confidence", 0.0)

        if not papers:
            return self._no_results_response(intent)

        context_str = self._format_contexts(papers)
        graph_section = self._format_graph_section(graph_triplets)

        system_prompt = (
            "You are a senior AI research analyst.\n"
            "The user wants a direct, structured COMPARISON between methods, models, or approaches.\n\n"
            "Format your answer as:\n"
            "1. A comparison table if comparing quantitative metrics\n"
            "2. A structured breakdown of differences in approach, performance, and limitations\n"
            "3. A clear verdict on which approach is stronger and under what conditions\n\n"
            "Always cite papers: [arXiv:ID - Short Title]\n"
            "Use graph relationships to link entities across papers."
        )

        human_prompt = (
            f"Comparison Question: {question}\n"
            f"{graph_section}"
            f"\nResearch Context:\n{context_str}\n\n"
            "Provide a structured comparison with citations."
        )

        answer = await self._generate(system_prompt, human_prompt)
        return self._build_response(answer, papers, intent.intent, confidence)

    # ------------------------------------------------------------------ #
    # TARGETED QUERY HANDLER
    # ------------------------------------------------------------------ #
    async def _handle_targeted_query(
        self, question: str, intent: QueryIntent, topic: Optional[str]
    ) -> Dict[str, Any]:
        logger.info(f"Targeted query ({intent.intent}) — expanded: {intent.expanded_query[:80]}")

        retrieved = await self.retriever.search(
            intent.expanded_query, topic=topic, n_results=8
        )
        papers = retrieved.get("papers", [])
        graph_triplets = retrieved.get("graph_triplets", [])
        confidence = retrieved.get("retrieval_confidence", 0.0)

        if not papers:
            return self._no_results_response(intent)

        context_str = self._format_contexts(papers)
        graph_section = self._format_graph_section(graph_triplets)

        confidence_warning = ""
        if confidence < 0.40:
            confidence_warning = (
                "\n\n⚠️ **Note**: Retrieval confidence is low. "
                "Consider ingesting more papers on this sub-topic."
            )

        system_prompt = (
            "You are a senior AI Research Engineer with access to a personal research knowledge base.\n\n"
            "Answer the user's question **strictly based on the provided context**.\n\n"
            "Rules:\n"
            "- Ground every claim in the provided context.\n"
            "- Always cite papers: [arXiv:ID - Short Title]\n"
            "- If context is insufficient, say so explicitly.\n"
            "- Be technical, precise, and insightful.\n"
            "- Use graph relationships to connect methods, datasets, and metrics.\n"
            "- Go beyond summaries — provide analysis, implications, and connections."
        )

        human_prompt = (
            f"Question: {question}\n"
            f"{graph_section}"
            f"\nResearch Context:\n{context_str}\n\n"
            "Answer using only the above context. Include citations."
        )

        answer = await self._generate(system_prompt, human_prompt)
        answer += confidence_warning
        return self._build_response(answer, papers, intent.intent, confidence)

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #
    async def _generate(self, system_prompt: str, human_prompt: str) -> str:
        try:
            response = await gateway.generate(
                task="research_answer",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt}
                ],
                temperature=0.2
            )
            return response.text
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"

    def _format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        parts = []
        for i, ctx in enumerate(contexts, 1):
            paper_id = ctx.get("paper_id", "unknown")
            title = ctx.get("title", "Untitled")
            score = ctx.get("score", 0.0)
            content = ctx.get("content") or ctx.get("text") or ""

            parts.append(
                f"--- Paper {i} [arXiv:{paper_id}] {title} [score: {score:.3f}] ---\n"
                f"{content.strip()}\n"
            )
        return "\n".join(parts)

    def _format_graph_section(self, graph_triplets: List[str]) -> str:
        if not graph_triplets:
            return ""
        lines = "\n".join(f"  - {t}" for t in graph_triplets[:15])
        return f"\nKnowledge Graph Relationships:\n{lines}\n"

    def _no_results_response(self, intent: QueryIntent) -> Dict[str, Any]:
        return {
            "answer": (
                "No relevant papers were found in the knowledge base for this query. "
                "This may mean: (1) no papers on this topic are indexed yet, or "
                "(2) the query is too specific for the current collection.\n\n"
                "Try ingesting more papers on this topic or broadening your query."
            ),
            "sources": [],
            "contexts_used": 0,
            "intent": intent.intent,
            "retrieval_confidence": 0.0
        }

    def _build_response(
        self,
        answer: str,
        papers: List[Dict],
        intent_type: str,
        confidence: float
    ) -> Dict[str, Any]:
        sources = [
            {
                "paper_id": c.get("paper_id"),
                "title": c.get("title", "Untitled"),
                "arxiv_url": c.get("arxiv_url", f"https://arxiv.org/abs/{c.get('paper_id')}"),
                "score": c.get("score"),
            }
            for c in papers
        ]
        return {
            "answer": answer,
            "sources": sources,
            "contexts_used": len(papers),
            "intent": intent_type,
            "retrieval_confidence": confidence
        }


query_agent = QueryAgent()
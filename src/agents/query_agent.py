"""
Query / Chat Agent - Intent-routed RAG with Paper Resolver.
"""

import re
from typing import List, Dict, Any, Optional
from loguru import logger
from src.gateway import gateway
from src.tools.retriever import research_retriever
from src.agents.intent_classifier import intent_classifier, QueryIntent
from src.agents.synthesis_agent import synthesis_agent
from src.db.neo4j_client import neo4j_client
from src.agents.session_manager import session_manager
from src.observability.tracing import traced


class QueryAgent:
    """Intent-routed RAG agent with paper-number resolution."""

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
    # PAPER RESOLVER
    # ------------------------------------------------------------------ #
    def resolve_paper_reference(self, question: str, paper_map: Dict[int, str]) -> Optional[str]:
        """
        Detects:
          - paper 1 / paper 3
          - the first paper / third paper
        and returns the corresponding arXiv ID.
        """
        if not paper_map:
            return None

        ordinals = {
            "first": 1, "second": 2, "third": 3,
            "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
        }

        q = question.lower()

        # paper 3 / paper 1
        m = re.search(r"paper\s+(\d+)", q)
        if m:
            num = int(m.group(1))
            return paper_map.get(num)

        # the first paper / third paper
        m = re.search(r"(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+paper", q)
        if m:
            num = ordinals.get(m.group(1))
            return paper_map.get(num)

        return None

    # ------------------------------------------------------------------ #
    # EXPAND COLLECTION
    # ------------------------------------------------------------------ #
    def _handle_expand_collection(self, question, intent, topic):
        return {
            "answer": (
                f"To fetch more papers for this session, use the `/ingest` command.\n\n"
                f"This will run the full arXiv search pipeline for **'{topic}'**."
            ),
            "sources": [],
            "contexts_used": 0,
            "intent": "expand_collection",
            "retrieval_confidence": 1.0
        }

    # ------------------------------------------------------------------ #
    # COLLECTION-LEVEL
    # ------------------------------------------------------------------ #
    async def _handle_collection_query(self, question, intent, topic):
        logger.info(f"Collection-level query ({intent.intent}) — loading grouped papers for '{topic}'")

        notes = await self.retriever.get_grouped_notes_for_topic(topic) if topic else []

        if not notes:
            return {
                "answer": f"No papers are indexed for the topic '{topic}' yet. Use `/ingest`.",
                "sources": [],
                "contexts_used": 0,
                "intent": intent.intent,
                "retrieval_confidence": 0.0
            }

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
    # COMPARISON
    # ------------------------------------------------------------------ #
    async def _handle_comparison_query(self, question, intent, topic):
        retrieved = await self.retriever.search(intent.expanded_query, topic=topic, n_results=8)
        papers = retrieved.get("papers", [])
        graph_triplets = retrieved.get("graph_triplets", [])
        confidence = retrieved.get("retrieval_confidence", 0.0)

        if not papers:
            return self._no_results_response(intent)

        context_str = self._format_contexts(papers)
        graph_section = self._format_graph_section(graph_triplets)

        system_prompt = (
            "You are a senior AI research analyst.\n"
            "Provide a structured COMPARISON. Always cite [arXiv:ID - Title]."
        )
        human_prompt = f"Comparison: {question}\n{graph_section}\nContext:\n{context_str}"

        answer = await self._generate(system_prompt, human_prompt)
        return self._build_response(answer, papers, intent.intent, confidence)

    # ------------------------------------------------------------------ #
    # TARGETED QUERY (with Paper Resolver)
    # ------------------------------------------------------------------ #
    async def _handle_targeted_query(self, question, intent, topic):
        logger.info(f"Targeted query ({intent.intent}) — expanded: {intent.expanded_query[:80]}")

        # --- Paper Resolver ---
        paper_map = session_manager.build_paper_number_map()
        resolved_id = self.resolve_paper_reference(question, paper_map)

        if resolved_id:
            logger.info(f"Resolved paper reference → {resolved_id}")
            # Fetch chunks only for this specific paper
            papers = await self.retriever.get_chunks_for_paper(resolved_id, topic=topic, n_results=50)
            if not papers:
                # Fallback to normal search if dedicated method not available
                retrieved = await self.retriever.search(
                    f"paper {resolved_id}", topic=topic, n_results=8
                )
                papers = [p for p in retrieved.get("papers", []) if p.get("paper_id") == resolved_id]

            graph_triplets = []
            confidence = 0.95 if papers else 0.0
        else:
            # Normal semantic search
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
            "- If context is insufficient, say so explicitly (this is NOT a hallucination).\n"
            "- Be technical, precise, and insightful."
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
                "No relevant papers were found in the knowledge base for this query.\n\n"
                "Try ingesting more papers or broadening your query."
            ),
            "sources": [],
            "contexts_used": 0,
            "intent": intent.intent,
            "retrieval_confidence": 0.0
        }

    def _build_response(self, answer, papers, intent_type, confidence):
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
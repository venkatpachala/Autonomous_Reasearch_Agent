"""
Query / Chat Agent - Talks to Helix Research using intent-routed RAG.

Pipeline:
  User Query
      ↓
  Intent Classifier  (7 intent types + query expansion)
      ↓
  Collection-level?  ──yes──> Load ALL notes → Synthesis Agent
      │ no
      ↓
  Vector Search (with expanded query + score threshold)
      ↓
  Graph RAG Enrichment
      ↓
  Grounded RAG Answer (with confidence warning if weak)
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
        """
        Main entry point. Classifies intent, routes to appropriate handler,
        and generates a grounded, cited, insight-rich answer.
        """

        # === STEP 1: Intent Classification + Query Expansion ===
        intent: QueryIntent = await intent_classifier.classify(question, topic=topic)

        # === STEP 2: Route by Intent ===
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
        """User wants to fetch more papers. Redirect to /ingest."""
        logger.info(f"Expand collection intent detected for topic '{topic}'")
        return {
            "answer": (
                f"To fetch more papers for this session, use the `/ingest` command.\n\n"
                f"This will run the full arXiv search pipeline for the current topic "
                f"**'{topic}'** and add new papers to your knowledge base.\n\n"
                "You can also type `/ingest <subtopic>` to search for a specific sub-area "
                "within this research topic."
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
        """Loads ALL notes for the topic and runs the SynthesisAgent."""
        logger.info(f"Collection-level query ({intent.intent}) — loading all notes for '{topic}'")

        notes = await self.retriever.get_all_notes_for_topic(topic) if topic else []

        if not notes:
            return {
                "answer": (
                    f"No papers are indexed for the topic '{topic}' yet. "
                    "Please ingest papers on this topic first using the Research button."
                ),
                "sources": [],
                "contexts_used": 0,
                "intent": intent.intent,
                "retrieval_confidence": 0.0
            }

        # Enrich with graph triplets from all concept entities
        all_concepts = []
        for note in notes:
            all_concepts.extend(note.concepts or [])
        all_concepts = list(set(all_concepts))[:30]

        graph_triplets = []
        if neo4j_client.is_connected() and all_concepts:
            graph_triplets = neo4j_client.get_related_triplets(all_concepts)

        result = await synthesis_agent.synthesize(
            notes=notes,
            query=question,
            topic=topic or "research collection",
            graph_triplets=graph_triplets
        )
        result["intent"] = intent.intent
        result["retrieval_confidence"] = 1.0  # Full collection — confidence is high
        return result

    # ------------------------------------------------------------------ #
    # COMPARISON HANDLER
    # ------------------------------------------------------------------ #
    async def _handle_comparison_query(
        self, question: str, intent: QueryIntent, topic: Optional[str]
    ) -> Dict[str, Any]:
        """Uses expanded query + comparison-specific prompt."""
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
    # TARGETED QUERY HANDLER (fact_lookup / paper_summary / general_qa)
    # ------------------------------------------------------------------ #
    async def _handle_targeted_query(
        self, question: str, intent: QueryIntent, topic: Optional[str]
    ) -> Dict[str, Any]:
        """Standard RAG: vector search on expanded query + grounded generation."""
        logger.info(
            f"Targeted query ({intent.intent}) — "
            f"expanded: {intent.expanded_query[:80]}"
        )

        # Use the expanded query for retrieval, not the raw question
        retrieved = await self.retriever.search(
            intent.expanded_query, topic=topic, n_results=6
        )
        papers = retrieved.get("papers", [])
        graph_triplets = retrieved.get("graph_triplets", [])
        confidence = retrieved.get("retrieval_confidence", 0.0)

        if not papers:
            return self._no_results_response(intent)

        context_str = self._format_contexts(papers)
        graph_section = self._format_graph_section(graph_triplets)

        confidence_warning = ""
        if confidence < 0.35:
            confidence_warning = (
                "\n\n⚠️ **Low retrieval confidence** — the retrieved papers may not be "
                "closely related to this specific question. Consider ingesting more relevant papers."
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
            note = ctx.get("full_note")
            title = ctx["title"]
            paper_id = ctx["paper_id"]
            content = ctx["content"]
            score_str = f"[relevance: {ctx['score']:.3f}]" if ctx.get("score") else ""

            extra = ""
            if note:
                extra = f"\nSummary: {note.one_sentence_summary}"
                if note.structured_data:
                    sd = note.structured_data
                    if sd.key_contributions:
                        extra += f"\nKey Contributions: {'; '.join(sd.key_contributions[:3])}"
                    if sd.limitations:
                        extra += f"\nLimitations: {'; '.join(sd.limitations[:2])}"
                    if sd.benchmarks:
                        extra += f"\nBenchmarks: {sd.benchmarks[:2]}"

            parts.append(
                f"--- Paper {i} {score_str}: [{paper_id}] {title} ---\n"
                f"{content}{extra}\n"
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
                "paper_id": c["paper_id"],
                "title": c["title"],
                "arxiv_url": c["arxiv_url"],
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

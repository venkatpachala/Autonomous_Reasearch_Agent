"""
Query / Chat Agent - Intent-routed RAG with Paper Resolver + Session Metadata.
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
from src.tools.research_index import research_index
from src.observability.tracing import traced


class QueryAgent:
    """Intent-routed RAG agent with paper-number resolution and metadata lookup."""

    def __init__(self):
        self.retriever = research_retriever

    @traced(name="query_agent_answer", run_type="chain")
    async def answer(
        self,
        question: str,
        topic: Optional[str] = None,
        chat_history: Optional[List] = None,
    ) -> Dict[str, Any]:
        # 1. Session metadata — NEVER use RAG
        if self._is_session_metadata_question(question):
            logger.info("Routing to session metadata lookup (no RAG)")
            return self._answer_session_metadata(question, topic)

        # 2. Normal intent routing
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
    # SESSION METADATA
    # ------------------------------------------------------------------ #
    def _is_session_metadata_question(self, question: str) -> bool:
        q = question.lower().strip()
        patterns = [
            r"how many papers",
            r"number of papers",
            r"count.*papers",
            r"list (all )?papers",
            r"list (all )?(arxiv )?ids",
            r"list (all )?titles",
            r"what papers",
            r"which papers",
            r"show (all )?papers",
            r"papers ingested",
            r"papers indexed",
            r"ingestion status",
            r"failed papers",
            r"session topic",
            r"current topic",
            r"what( is|\'s)? (the )?topic",
            r"this session",
        ]
        return any(re.search(p, q) for p in patterns)

    def _answer_session_metadata(
        self, question: str, topic: Optional[str]
    ) -> Dict[str, Any]:
        session = session_manager.current_session
        paper_ids = (session.papers_ingested if session else []) or []
        session_topic = topic or (session.topic if session else None)

        rows = []
        for i, pid in enumerate(paper_ids, 1):
            meta = research_index.get_paper(pid) or {}
            rows.append({
                "n": i,
                "paper_id": pid,
                "title": meta.get("title") or "Untitled",
                "authors": meta.get("authors") or [],
            })

        q = question.lower()

        if re.search(r"session topic|current topic|what( is|\'s)? (the )?topic", q):
            answer = f"**Session topic:** {session_topic or 'N/A'}"
            if paper_ids:
                answer += f"\n**Papers ingested:** {len(paper_ids)}"

        elif re.search(r"how many|number of|count", q):
            answer = f"**{len(paper_ids)} papers** are ingested in this session"
            if session_topic:
                answer += f" for topic **'{session_topic}'**."
            else:
                answer += "."
            if rows:
                answer += "\n\n" + "\n".join(
                    f"{r['n']}. `{r['paper_id']}` — {r['title']}" for r in rows
                )

        elif re.search(r"list.*id|arxiv", q):
            if not paper_ids:
                answer = "No papers are ingested in this session yet."
            else:
                answer = "**arXiv IDs in this session:**\n" + "\n".join(
                    f"{r['n']}. `{r['paper_id']}`" for r in rows
                )

        elif re.search(
            r"list.*title|what papers|which papers|show.*papers|list.*papers", q
        ):
            if not rows:
                answer = "No papers are ingested in this session yet."
            else:
                answer = f"**Papers in this session ({len(rows)}):**\n" + "\n".join(
                    f"{r['n']}. **{r['title']}** (`{r['paper_id']}`)" for r in rows
                )

        else:
            answer = (
                f"**Session status**\n"
                f"- Topic: {session_topic or 'N/A'}\n"
                f"- Papers ingested: {len(paper_ids)}\n"
            )
            if rows:
                answer += "\n" + "\n".join(
                    f"{r['n']}. {r['title']} (`{r['paper_id']}`)" for r in rows
                )

        sources = [
            {
                "paper_id": r["paper_id"],
                "title": r["title"],
                "arxiv_url": f"https://arxiv.org/abs/{r['paper_id']}",
                "score": 1.0,
            }
            for r in rows
        ]

        return {
            "answer": answer,
            "sources": sources,
            "contexts_used": len(rows),
            "intent": "metadata_lookup",
            "retrieval_confidence": 1.0,
        }

    # ------------------------------------------------------------------ #
    # PAPER RESOLVER
    # ------------------------------------------------------------------ #
    def resolve_paper_reference(
        self, question: str, paper_map: Dict[int, str]
    ) -> Optional[str]:
        if not paper_map:
            return None

        ordinals = {
            "first": 1, "second": 2, "third": 3,
            "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        }

        q = question.lower()

        m = re.search(r"paper\s+(\d+)", q)
        if m:
            return paper_map.get(int(m.group(1)))

        m = re.search(
            r"(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+paper",
            q,
        )
        if m:
            return paper_map.get(ordinals.get(m.group(1)))

        return None

    def resolve_arxiv_id(self, question: str) -> Optional[str]:
        m = re.search(r"\b(\d{4}\.\d{4,5}v?\d*)\b", question)
        return m.group(1) if m else None

    # ------------------------------------------------------------------ #
    # PAPER METADATA
    # ------------------------------------------------------------------ #
    def _is_paper_metadata_question(self, question: str) -> bool:
        q = question.lower()
        keywords = [
            "author", "authors", "who wrote", "written by",
            "published", "publication date", "year",
            "affiliation", "venue", "category", "categories",
            "what is the title", "title of paper",
            "abstract of",
        ]
        return any(k in q for k in keywords)

    def _answer_from_paper_metadata(
        self, question: str, paper_id: str
    ) -> Dict[str, Any]:
        meta = research_index.get_paper(paper_id)
        if not meta:
            return {
                "answer": (
                    f"No metadata stored yet for `{paper_id}`. "
                    "Re-ingest to populate authors/dates."
                ),
                "sources": [],
                "contexts_used": 0,
                "intent": "metadata",
                "retrieval_confidence": 0.0,
            }

        q = question.lower()
        title = meta.get("title", "Untitled")
        authors = meta.get("authors") or []
        published = meta.get("published") or "Unknown"
        abstract = meta.get("abstract") or ""
        categories = meta.get("categories") or []

        if any(k in q for k in ["author", "who wrote", "written by"]):
            if authors:
                answer = f"**Authors of [{paper_id}] {title}:**\n" + ", ".join(authors)
            else:
                answer = (
                    f"Author information for [{paper_id}] {title} "
                    "is not in the metadata registry yet. Re-ingest to populate authors."
                )
        elif any(k in q for k in ["published", "year", "publication date"]):
            answer = f"**Published:** {published}\n**Paper:** [{paper_id}] {title}"
        elif "title" in q:
            answer = f"**Title:** {title}\n**arXiv:** {paper_id}"
        elif "categor" in q:
            cats = ", ".join(categories) if categories else "Not available"
            answer = f"**Categories:** {cats}\n**Paper:** [{paper_id}] {title}"
        elif "abstract" in q:
            answer = (
                f"**Abstract of [{paper_id}] {title}:**\n\n"
                f"{abstract or 'Not available.'}"
            )
        else:
            author_str = ", ".join(authors) if authors else "Not available"
            answer = (
                f"**[{paper_id}] {title}**\n\n"
                f"- **Authors:** {author_str}\n"
                f"- **Published:** {published}\n"
                f"- **Categories:** {', '.join(categories) if categories else 'N/A'}\n"
                f"- **arXiv:** https://arxiv.org/abs/{paper_id}\n"
            )
            if abstract:
                answer += (
                    f"\n**Abstract:**\n"
                    f"{abstract[:800]}{'...' if len(abstract) > 800 else ''}"
                )

        return {
            "answer": answer,
            "sources": [{
                "paper_id": paper_id,
                "title": title,
                "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
                "score": 1.0,
            }],
            "contexts_used": 1,
            "intent": "metadata",
            "retrieval_confidence": 1.0,
        }

    # ------------------------------------------------------------------ #
    # EXPAND / COLLECTION / COMPARISON / TARGETED
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
            "retrieval_confidence": 1.0,
        }

    async def _handle_collection_query(self, question, intent, topic):
        logger.info(
            f"Collection-level query ({intent.intent}) — "
            f"loading grouped papers for '{topic}'"
        )

        notes = await self.retriever.get_grouped_notes_for_topic(topic) if topic else []

        if not notes:
            return {
                "answer": (
                    f"No papers are indexed for the topic '{topic}' yet. Use `/ingest`."
                ),
                "sources": [],
                "contexts_used": 0,
                "intent": intent.intent,
                "retrieval_confidence": 0.0,
            }

        graph_triplets = []
        if neo4j_client.is_connected() and notes:
            paper_ids = list({n.get("paper_id") for n in notes if n.get("paper_id")})
            graph_triplets = neo4j_client.get_related_triplets(paper_ids[:20])

        result = await synthesis_agent.synthesize(
            notes=notes,
            query=question,
            topic=topic or "research collection",
            graph_triplets=graph_triplets,
        )
        result["intent"] = intent.intent
        result["retrieval_confidence"] = 1.0
        return result

    async def _handle_comparison_query(self, question, intent, topic):
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
            "Provide a structured COMPARISON. Always cite [arXiv:ID - Title]."
        )
        human_prompt = f"Comparison: {question}\n{graph_section}\nContext:\n{context_str}"

        answer = await self._generate(system_prompt, human_prompt)
        return self._build_response(answer, papers, intent.intent, confidence)

    async def _handle_targeted_query(self, question, intent, topic):
        logger.info(
            f"Targeted query ({intent.intent}) — expanded: {intent.expanded_query[:80]}"
        )

        paper_map = session_manager.build_paper_number_map()
        resolved_id = (
            self.resolve_paper_reference(question, paper_map)
            or self.resolve_arxiv_id(question)
        )

        if resolved_id and self._is_paper_metadata_question(question):
            logger.info(f"Answered from paper metadata registry: {resolved_id}")
            return self._answer_from_paper_metadata(question, resolved_id)

        if resolved_id:
            logger.info(f"Resolved paper reference → {resolved_id}")
            papers = await self.retriever.get_chunks_for_paper(
                resolved_id, topic=topic, n_results=50
            )
            if not papers:
                retrieved = await self.retriever.search(
                    f"paper {resolved_id}", topic=topic, n_results=8
                )
                papers = [
                    p for p in retrieved.get("papers", [])
                    if p.get("paper_id") == resolved_id
                ]
            graph_triplets = []
            confidence = 0.95 if papers else 0.0
        else:
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
            "You are a senior AI Research Engineer with access to a personal "
            "research knowledge base.\n\n"
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
                    {"role": "user", "content": human_prompt},
                ],
                temperature=0.2,
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
            "retrieval_confidence": 0.0,
        }

    def _build_response(self, answer, papers, intent_type, confidence):
        sources = [
            {
                "paper_id": c.get("paper_id"),
                "title": c.get("title", "Untitled"),
                "arxiv_url": c.get(
                    "arxiv_url", f"https://arxiv.org/abs/{c.get('paper_id')}"
                ),
                "score": c.get("score"),
            }
            for c in papers
        ]
        return {
            "answer": answer,
            "sources": sources,
            "contexts_used": len(papers),
            "intent": intent_type,
            "retrieval_confidence": confidence,
        }


query_agent = QueryAgent()
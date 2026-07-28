"""
Query / Chat Agent - Intent-routed RAG with Paper Resolver + Session Metadata.
Instrumented with per-stage latency logging for interactive path profiling.
"""

import re
import time
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

    async def answer(
        self,
        question: str,
        topic: Optional[str] = None,
        chat_history: Optional[List] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        stages: Dict[str, float] = {}

        def mark(name: str, t_start: float) -> None:
            stages[name] = (time.perf_counter() - t_start) * 1000.0

        # ------------------------------------------------------------------ #
        # 1. Session metadata — zero LLM
        # ------------------------------------------------------------------ #
        t = time.perf_counter()
        if self._is_session_metadata_question(question):
            logger.info("Routing to session metadata lookup (no RAG)")
            out = self._answer_session_metadata(question, topic)
            mark("metadata_ms", t)
            stages["total_ms"] = (time.perf_counter() - t0) * 1000.0
            logger.info(
                f"Query latency ms | path=metadata "
                f"metadata={stages.get('metadata_ms', 0):.0f} "
                f"total={stages['total_ms']:.0f}"
            )
            out["latency_ms"] = stages
            return out

        # ------------------------------------------------------------------ #
        # 2. Out-of-range paper N / page count — never RAG (Bug 3)
        # ------------------------------------------------------------------ #
        guarded = self._answer_out_of_range_or_pages(question, topic)
        if guarded is not None:
            logger.info(
                f"Routing to ordinal/page guard (intent={guarded.get('intent')})"
            )
            stages["intent_ms"] = 0.0
            stages["retrieve_ms"] = 0.0
            stages["llm_ms"] = 0.0
            stages["total_ms"] = (time.perf_counter() - t0) * 1000.0
            guarded["latency_ms"] = stages
            logger.info(
                f"Query latency ms | path={guarded.get('intent')} "
                f"total={stages['total_ms']:.0f}"
            )
            return guarded

        # ------------------------------------------------------------------ #
        # 3. Paper ordinal / arXiv id resolution (no intent LLM yet)
        # ------------------------------------------------------------------ #
        paper_map = session_manager.build_paper_number_map()
        resolved_id = (
            self.resolve_paper_reference(question, paper_map)
            or self.resolve_arxiv_id(question)
        )

        # 3a. Authors / title / year / abstract → registry only
        if resolved_id and self._is_paper_metadata_question(question):
            logger.info(f"Fast path: paper metadata registry → {resolved_id}")
            out = self._answer_from_paper_metadata(question, resolved_id)
            stages["intent_ms"] = 0.0
            stages["retrieve_ms"] = 0.0
            stages["llm_ms"] = 0.0
            stages["total_ms"] = (time.perf_counter() - t0) * 1000.0
            logger.info(
                f"Query latency ms | path=metadata "
                f"intent=0 retrieve=0 llm=0 total={stages['total_ms']:.0f}"
            )
            out["latency_ms"] = stages
            return out

        # 3b. "Describe / summarise paper N" → chunks + one answer LLM (skip intent)
        if resolved_id and self._is_paper_describe_question(question):
            logger.info(f"Fast path: describe paper → {resolved_id} (skip intent)")
            out = await self._answer_paper_describe(
                question, resolved_id, topic, stages
            )
            stages["total_ms"] = (time.perf_counter() - t0) * 1000.0
            logger.info(
                f"Query latency ms | path=paper_describe "
                f"intent=0 "
                f"retrieve={stages.get('retrieve_ms', 0):.0f} "
                f"llm={stages.get('llm_ms', 0):.0f} "
                f"total={stages['total_ms']:.0f}"
            )
            out["latency_ms"] = stages
            return out
    # ------------------------------------------------------------------ #
    # 3. Intent: rules first (<5ms), LLM only if needed
    # ------------------------------------------------------------------ #
        t = time.perf_counter()
        cheap = self._cheap_intent(question, topic)
        if cheap is not None:
            intent = cheap
            stages["intent_ms"] = (time.perf_counter() - t) * 1000.0  # should be ~0
            logger.info(
                f"Cheap intent: {intent.intent} [{intent.confidence}] "
                f"in {stages['intent_ms']:.1f}ms — skip LLM classifier"
            )
        else:
            intent = await intent_classifier.classify(question, topic=topic)
            stages["intent_ms"] = (time.perf_counter() - t) * 1000.0
            logger.info(
                f"LLM intent: {intent.intent} [{getattr(intent, 'confidence', '?')}] "
                f"in {stages['intent_ms']:.0f}ms"
            )

        if intent_classifier.is_collection_level(intent):
            out = await self._handle_collection_query(
                question, intent, topic, stages
            )
        elif intent.intent == "comparison":
            out = await self._handle_comparison_query(
                question, intent, topic, stages
            )
        elif intent.intent == "expand_collection":
            out = self._handle_expand_collection(question, intent, topic)
        else:
            out = await self._handle_targeted_query(
                question, intent, topic, stages, resolved_id=resolved_id
            )

        if not out or not isinstance(out, dict):
            logger.error(
                f"Handler returned invalid result: type={type(out)} value={out!r}"
            )
            out = {
                "answer": "Internal error: empty handler result.",
                "sources": [],
                "contexts_used": 0,
                "intent": getattr(intent, "intent", "unknown"),
                "retrieval_confidence": 0.0,
            }

        stages["total_ms"] = (time.perf_counter() - t0) * 1000.0
        out["latency_ms"] = stages
        logger.info(
            f"Query latency ms | path={out.get('intent', '?')} "
            f"intent={stages.get('intent_ms', 0):.0f} "
            f"retrieve={stages.get('retrieve_ms', 0):.0f} "
            f"llm={stages.get('llm_ms', 0):.0f} "
            f"synth={stages.get('synth_ms', 0):.0f} "
            f"total={stages['total_ms']:.0f}"
        )
        return out   # ← must exist; this was almost certainly missing
    
    def _cheap_intent(
        self, question: str, topic: Optional[str] = None
    ) -> Optional["QueryIntent"]:
        """
        Deterministic intent in <5ms.
        Returns QueryIntent when confident; None → fall back to LLM classifier.
        """
        q = question.lower().strip()

        # --- collection / overview ---
        if re.search(
            r"\b("
            r"summarise all|summarize all|summary of all|"
            r"overview of (the )?(papers|collection|research)|"
            r"all (the )?papers|entire collection|"
            r"literature review|what are (all )?these papers|"
            r"across (the )?papers|research directions|"
            r"state of the art|trends and gaps"
            r")\b",
            q,
        ):
            return QueryIntent(
                intent="collection_overview",
                confidence=0.92,
                expanded_query=question,
                reasoning="rule: collection_overview",
            )

        # --- comparison ---
        if re.search(
            r"\b(compare|comparison|versus|\bvs\.?\b|difference between|how does .+ differ)\b",
            q,
        ):
            return QueryIntent(
                intent="comparison",
                confidence=0.9,
                expanded_query=question,
                reasoning="rule: comparison",
            )

        # --- expand collection ---
        if re.search(
            r"\b(more papers|fetch more|ingest more|expand (the )?collection|find more papers)\b",
            q,
        ):
            return QueryIntent(
                intent="expand_collection",
                confidence=0.95,
                expanded_query=question,
                reasoning="rule: expand_collection",
            )

        # --- fact / semantic RAG (methods, how, what, benchmarks, ...) ---
        if re.search(
            r"\b("
            r"method|methods|technique|techniques|approach|approaches|"
            r"benchmark|benchmarks|metric|metrics|dataset|datasets|"
            r"architecture|architectures|model|models|"
            r"result|results|limitation|limitations|ablation|"
            r"algorithm|algorithms|framework|frameworks|"
            r"how (do|does|is|are|can|to)|"
            r"what (is|are|do|does|was|were)|"
            r"why (is|are|do|does)|"
            r"which (method|model|approach|technique)|"
            r"explain|list the|used for|used in"
            r")\b",
            q,
        ):
            return QueryIntent(
                intent="fact_lookup",
                confidence=0.88,
                expanded_query=question,  # hybrid+rerank don't need LLM rewrite
                reasoning="rule: fact_lookup",
            )

        # --- single-paper summary without ordinal already handled in fast path ---
        if re.search(
            r"\b(paper_summary|summarise paper|summarize paper|describe the paper)\b",
            q,
        ):
            return QueryIntent(
                intent="paper_summary",
                confidence=0.85,
                expanded_query=question,
                reasoning="rule: paper_summary",
            )

        return None

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

    async def _handle_collection_query(
        self, question, intent, topic, stages: Optional[Dict] = None
    ):
        stages = stages if stages is not None else {}
        logger.info(
            f"Collection-level query ({intent.intent}) — "
            f"loading grouped papers for '{topic}'"
        )

        t = time.perf_counter()
        notes = await self.retriever.get_grouped_notes_for_topic(topic) if topic else []
        stages["retrieve_ms"] = (time.perf_counter() - t) * 1000.0

        if not notes:
            return {
                "answer": (
                    f"No papers are indexed for the topic '{topic}' yet. Use `/ingest`."
                ),
                "sources": [],
                "contexts_used": 0,
                "intent": intent.intent,
                "retrieval_confidence": 0.0,
                "latency_ms": stages,
            }

        graph_triplets = []
        try:
            if neo4j_client.is_connected() and notes:
                paper_ids = list(
                    {n.get("paper_id") for n in notes if n.get("paper_id")}
                )
                graph_triplets = neo4j_client.get_related_triplets(paper_ids[:20]) or []
        except Exception as e:
            logger.warning(f"Graph enrichment skipped: {e}")

        t = time.perf_counter()
        try:
            raw = await synthesis_agent.synthesize(
                notes=notes,
                query=question,
                topic=topic or "research collection",
                graph_triplets=graph_triplets,
            )
        except Exception as e:
            logger.exception(f"synthesis_agent.synthesize failed: {e}")
            stages["synth_ms"] = (time.perf_counter() - t) * 1000.0
            return {
                "answer": f"Synthesis failed: {e}",
                "sources": [],
                "contexts_used": 0,
                "intent": intent.intent,
                "retrieval_confidence": 0.0,
                "latency_ms": stages,
            }
        stages["synth_ms"] = (time.perf_counter() - t) * 1000.0
        stages["llm_ms"] = stages["synth_ms"]

        # --- Normalize synthesize() output (must always return a chat dict) ---
        def _sources_from_notes():
            return [
                {
                    "paper_id": n.get("paper_id"),
                    "title": n.get("title", "Untitled"),
                    "arxiv_url": n.get(
                        "arxiv_url", f"https://arxiv.org/abs/{n.get('paper_id')}"
                    ),
                    "score": n.get("score"),
                }
                for n in notes
                if n.get("paper_id")
            ]

        if raw is None:
            logger.error("synthesis_agent.synthesize returned None")
            return {
                "answer": "Synthesis completed but returned no content.",
                "sources": _sources_from_notes(),
                "contexts_used": len(notes),
                "intent": intent.intent,
                "retrieval_confidence": 0.0,
                "latency_ms": stages,
            }

        if isinstance(raw, str):
            return {
                "answer": raw,
                "sources": _sources_from_notes(),
                "contexts_used": len(notes),
                "intent": intent.intent,
                "retrieval_confidence": 1.0,
                "latency_ms": stages,
            }

        if isinstance(raw, dict):
            answer = (
                raw.get("answer")
                or raw.get("text")
                or raw.get("synthesis")
                or raw.get("markdown")
            )
            # Cached path sometimes stores structured fields without "answer"
            if not answer and any(
                k in raw
                for k in ("research_directions", "key_methods", "state_of_the_art")
            ):
                parts = []
                if raw.get("research_directions"):
                    parts.append("### Research Directions")
                    for i, d in enumerate(raw["research_directions"], 1):
                        parts.append(f"{i}. {d}")
                if raw.get("key_methods"):
                    parts.append("\n### Key Methods")
                    for m in raw["key_methods"]:
                        parts.append(f"- {m}")
                if raw.get("emerging_trends"):
                    parts.append("\n### Emerging Trends")
                    for tr in raw["emerging_trends"]:
                        parts.append(f"- {tr}")
                if raw.get("research_gaps") or raw.get("gaps"):
                    gaps = raw.get("research_gaps") or raw.get("gaps")
                    parts.append("\n### Research Gaps")
                    for g in gaps:
                        parts.append(f"- {g}")
                if raw.get("state_of_the_art"):
                    parts.append(f"\n### State of the Art\n{raw['state_of_the_art']}")
                answer = "\n".join(parts) if parts else str(raw)

            sources = raw.get("sources") or _sources_from_notes()
            return {
                "answer": answer or "Synthesis produced no text.",
                "sources": sources,
                "contexts_used": raw.get("contexts_used", len(notes)),
                "intent": intent.intent,
                "retrieval_confidence": float(raw.get("retrieval_confidence", 1.0)),
                "latency_ms": stages,
            }

        # Pydantic ResearchSynthesis (or similar)
        if hasattr(raw, "model_dump"):
            data = raw.model_dump()
            formatter = getattr(synthesis_agent, "format_markdown", None) or getattr(
                synthesis_agent, "_format_markdown", None
            ) or getattr(synthesis_agent, "to_markdown", None)
            if callable(formatter):
                try:
                    answer = formatter(raw)
                except Exception as e:
                    logger.warning(f"format_markdown failed: {e}")
                    answer = str(data)
            else:
                # minimal fallback from structured fields
                answer = str(data)
            return {
                "answer": answer,
                "sources": _sources_from_notes(),
                "contexts_used": len(notes),
                "intent": intent.intent,
                "retrieval_confidence": 1.0,
                "latency_ms": stages,
            }

        logger.error(f"Unexpected synthesize return type: {type(raw)}")
        return {
            "answer": f"Unexpected synthesis result type: {type(raw).__name__}",
            "sources": _sources_from_notes(),
            "contexts_used": len(notes),
            "intent": intent.intent,
            "retrieval_confidence": 0.0,
            "latency_ms": stages,
        }

    async def _handle_comparison_query(
        self, question, intent, topic, stages: Optional[Dict] = None
    ):
        stages = stages if stages is not None else {}

        t = time.perf_counter()
        retrieved = await self.retriever.search(
            intent.expanded_query, topic=topic, n_results=8
        )
        stages["retrieve_ms"] = (time.perf_counter() - t) * 1000.0
        if retrieved.get("timings_ms"):
            stages["retrieve_detail"] = retrieved["timings_ms"]

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

        t = time.perf_counter()
        answer = await self._generate(system_prompt, human_prompt)
        stages["llm_ms"] = (time.perf_counter() - t) * 1000.0

        return self._build_response(answer, papers, intent.intent, confidence)
    def _extract_paper_ordinals(self, question: str) -> List[int]:
        """All 'paper N' / 'paper #N' numbers in the question."""
        return [int(m) for m in re.findall(
            r"\bpapers?\s*#?\s*(\d+)\b", question.lower()
        )]

    def _session_paper_count(self) -> int:
        session = session_manager.current_session
        if not session or not session.papers_ingested:
            return 0
        return len(session.papers_ingested)

    def _is_page_count_question(self, question: str) -> bool:
        q = question.lower()
        return bool(re.search(
            r"\b(how many pages|page count|number of pages|pages in)\b", q
        ))

    def _answer_out_of_range_or_pages(
        self, question: str, topic: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Return a response dict if we can answer without RAG; else None.
        Handles:
          - paper N when N > session paper count
          - page-count questions (not stored in index)
        """
        count = self._session_paper_count()
        ordinals = self._extract_paper_ordinals(question)

        # Out-of-range ordinal(s)
        if ordinals and count > 0:
            bad = [n for n in ordinals if n < 1 or n > count]
            if bad:
                bad_s = ", ".join(str(n) for n in bad)
                return {
                    "answer": (
                        f"This session only has **{count}** ingested papers "
                        f"(numbered 1–{count}). "
                        f"There is no paper {bad_s}.\n\n"
                        f"Use `/papers` or ask *What are titles of all papers ingested* "
                        f"to see the list."
                    ),
                    "sources": [],
                    "contexts_used": 0,
                    "intent": "paper_ordinal_guard",
                    "retrieval_confidence": 1.0,
                }

        # Page count — not in chunk metadata
        if self._is_page_count_question(question):
            # If they named a valid paper, point at the PDF
            resolved = None
            paper_map = session_manager.build_paper_number_map()
            if ordinals and count > 0:
                n = ordinals[0]
                if 1 <= n <= count:
                    resolved = paper_map.get(n)

            if resolved:
                meta = {}
                try:
                    meta = research_index.get_paper(resolved) or {}
                except Exception:
                    pass
                title = meta.get("title") or resolved
                return {
                    "answer": (
                        f"**Page count is not stored** in the research index for "
                        f"[arXiv:{resolved}] {title}.\n\n"
                        f"Open the PDF: https://arxiv.org/pdf/{resolved}\n"
                        f"or the abstract page: https://arxiv.org/abs/{resolved}"
                    ),
                    "sources": [{
                        "paper_id": resolved,
                        "title": title,
                        "arxiv_url": f"https://arxiv.org/abs/{resolved}",
                        "score": 1.0,
                    }],
                    "contexts_used": 1,
                    "intent": "page_count_guard",
                    "retrieval_confidence": 1.0,
                }

            return {
                "answer": (
                    "Page counts are **not indexed** in this system "
                    "(only parsed text chunks + metadata). "
                    "Open the paper PDF on arXiv to see the page count, "
                    "or ask about content (methods, results, authors)."
                ),
                "sources": [],
                "contexts_used": 0,
                "intent": "page_count_guard",
                "retrieval_confidence": 1.0,
            }

        return None
    async def _handle_targeted_query(
        self,
        question,
        intent,
        topic,
        stages: Optional[Dict] = None,
        resolved_id: Optional[str] = None,
    ):
        stages = stages if stages is not None else {}
        logger.info(
            f"Targeted query ({intent.intent}) — expanded: {intent.expanded_query[:80]}"
        )

        if resolved_id is None:
            paper_map = session_manager.build_paper_number_map()
            resolved_id = (
                self.resolve_paper_reference(question, paper_map)
                or self.resolve_arxiv_id(question)
            )

        if resolved_id and self._is_paper_metadata_question(question):
            logger.info(f"Answered from paper metadata registry: {resolved_id}")
            return self._answer_from_paper_metadata(question, resolved_id)

        t = time.perf_counter()
        if resolved_id:
            logger.info(f"Resolved paper reference → {resolved_id}")
            papers = await self.retriever.get_chunks_for_paper(
                resolved_id, topic=topic, n_results=8  # was 50 — was bloating llm
            )
            if not papers:
                retrieved = await self.retriever.search(
                    f"paper {resolved_id}", topic=topic, n_results=8
                )
                if retrieved.get("timings_ms"):
                    stages["retrieve_detail"] = retrieved["timings_ms"]
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
            if retrieved.get("timings_ms"):
                stages["retrieve_detail"] = retrieved["timings_ms"]
            papers = retrieved.get("papers", [])
            graph_triplets = retrieved.get("graph_triplets", [])
            confidence = retrieved.get("retrieval_confidence", 0.0)
        stages["retrieve_ms"] = (time.perf_counter() - t) * 1000.0

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
            "- If context is insufficient, say so explicitly.\n"
            "- Be technical, precise, and insightful."
        )

        human_prompt = (
            f"Question: {question}\n"
            f"{graph_section}"
            f"\nResearch Context:\n{context_str}\n\n"
            "Answer using only the above context. Include citations."
        )

        t = time.perf_counter()
        answer = await self._generate(system_prompt, human_prompt)
        stages["llm_ms"] = (time.perf_counter() - t) * 1000.0

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

    async def _generate_stream(
        self,
        system_prompt: str,
        human_prompt: str,
    ):
        """Async generator of token strings for chat streaming."""
        from src.gateway.streaming import stream_chat_tokens

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]
        async for delta in stream_chat_tokens(messages, temperature=0.2):
            yield delta
    def _is_paper_describe_question(self, question: str) -> bool:
        """Describe / summarise a single paper — skip intent classifier."""
        q = question.lower().strip()
        patterns = [
            r"\bdescribe\b",
            r"\bsummarise\b",
            r"\bsummarize\b",
            r"\bsummary of\b",
            r"\boverview of\b",
            r"\btell me about\b",
            r"\bwhat is paper\b",
            r"\bwhat'?s paper\b",
            r"\bexplain paper\b",
            r"\bdetailed description\b",
        ]
        return any(re.search(p, q) for p in patterns)

    async def _answer_paper_describe(
        self,
        question: str,
        paper_id: str,
        topic: Optional[str],
        stages: Dict[str, float],
    ) -> Dict[str, Any]:
        """Retrieve capped chunks for one paper + single answer LLM."""
        t = time.perf_counter()
        # Cap context: 8 chunks is enough for a solid summary and cuts llm_ms
        papers = await self.retriever.get_chunks_for_paper(
            paper_id, topic=topic, n_results=8
        )
        stages["retrieve_ms"] = (time.perf_counter() - t) * 1000.0

        if not papers:
            retrieved = await self.retriever.search(
                f"paper {paper_id}", topic=topic, n_results=8
            )
            if retrieved.get("timings_ms"):
                stages["retrieve_detail"] = retrieved["timings_ms"]
            papers = [
                p for p in retrieved.get("papers", [])
                if p.get("paper_id") == paper_id
            ]
            stages["retrieve_ms"] = (time.perf_counter() - t) * 1000.0

        if not papers:
            return {
                "answer": (
                    f"No indexed chunks found for `{paper_id}`. "
                    "Try `/ingest` or another paper number."
                ),
                "sources": [],
                "contexts_used": 0,
                "intent": "paper_describe",
                "retrieval_confidence": 0.0,
            }

        context_str = self._format_contexts(papers)
        system_prompt = (
            "You are a senior AI Research Engineer.\n"
            "Summarise the paper strictly from the provided context.\n"
            "Cover: problem, method, key results, limitations if present.\n"
            "Cite as [arXiv:ID - Short Title]. Be technical and concise."
        )
        human_prompt = (
            f"Question: {question}\n\n"
            f"Research Context:\n{context_str}\n\n"
            "Write a clear description of this paper only."
        )

        t = time.perf_counter()
        answer = await self._generate(system_prompt, human_prompt)
        stages["llm_ms"] = (time.perf_counter() - t) * 1000.0

        return self._build_response(answer, papers, "paper_describe", 0.95)
    
    async def _answer_paper_describe_stream(
        self,
        question: str,
        paper_id: str,
        topic: Optional[str],
        stages: Dict[str, float],
    ):
        """Yields ('token', str) then ('final', dict)."""
        t = time.perf_counter()
        papers = await self.retriever.get_chunks_for_paper(
            paper_id, topic=topic, n_results=8
        )
        stages["retrieve_ms"] = (time.perf_counter() - t) * 1000.0

        if not papers:
            yield (
                "final",
                {
                    "answer": f"No chunks indexed for paper {paper_id}.",
                    "sources": [],
                    "contexts_used": 0,
                    "intent": "paper_describe",
                    "retrieval_confidence": 0.0,
                    "latency_ms": stages,
                },
            )
            return

        context_str = self._format_contexts(papers)
        system_prompt = (
            "You are a senior AI Research Engineer.\n"
            "Answer strictly from the provided context. Cite [arXiv:ID - Title]."
        )
        human_prompt = (
            f"Question: {question}\n\nResearch Context:\n{context_str}\n\n"
            "Write a detailed technical description of this paper."
        )

        t = time.perf_counter()
        parts: List[str] = []
        async for delta in self._generate_stream(system_prompt, human_prompt):
            parts.append(delta)
            yield ("token", delta)
        stages["llm_ms"] = (time.perf_counter() - t) * 1000.0

        answer = "".join(parts) or "(empty answer)"
        out = self._build_response(
            answer, papers, "paper_describe", 0.95
        )
        out["latency_ms"] = stages
        yield ("final", out)
    
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
        best = {}
        for s in sources:
            pid = s.get("paper_id")
            if not pid:
                continue
            if pid not in best or (s.get("score") or 0) > (best[pid].get("score") or 0):
                best[pid] = s
        sources = list(best.values())
        return {
            "answer": answer,
            "sources": sources,
            "contexts_used": len(papers),
            "intent": intent_type,
            "retrieval_confidence": confidence,
        }


query_agent = QueryAgent()
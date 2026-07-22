"""
Phase-by-Phase Ingestion Pipeline Inspector
============================================
This script runs each phase of the paper fetching and ingestion pipeline step-by-step
and prints detailed inspection logs for each phase:

Phase 1: Keyword Decomposition (DecomposerAgent)
Phase 2: arXiv Paper Search & Fetch (arXiv Tool)
Phase 3: Relevance Filtering (RelevanceFilterAgent)
Phase 4: Single Paper PDF Extraction (PDFTools / pdf_extractor)
Phase 5: Summarization & Critic Note Generation (Summarizer & Critic Agents)
Phase 6: Layered Storage Persistence (MemoryManager -> Disk & Chroma)
Phase 7: Graph Triplet Extraction & Neo4j Insertion (ExtractorAgent & Neo4j)

Usage:
  .venv\\Scripts\\python.exe debug\\test_ingestion_phases.py
"""

import asyncio
import sys
import time
from typing import List
from loguru import logger

# Import all agents and tools
from src.agents.decomposer import decomposer_agent
from src.tools.arxiv_tool import arxiv_tool
from src.agents.relevance_filter import relevance_filter_agent
from src.agents.pdf_extractor import pdf_extractor_node
from src.agents.summarizer import summarizer_agent
from src.agents.critic_note import critic_agent
from src.agents.memory_manager import memory_manager
from src.agents.extractor_agent import extractor_agent
from src.db.neo4j_client import neo4j_client
from src.models.schemas import PaperMetadata


def print_banner(phase_num: int, title: str):
    print("\n" + "=" * 75)
    print(f"  PHASE {phase_num}: {title}")
    print("=" * 75)


async def test_pipeline_phases(topic: str):
    print("\n" + "#" * 75)
    print(f"  STARTING STEP-BY-STEP INGESTION INSPECTOR FOR TOPIC:")
    print(f"  '{topic}'")
    print("#" * 75)

    # -------------------------------------------------------------------------
    # PHASE 1: KEYWORD DECOMPOSITION
    # -------------------------------------------------------------------------
    print_banner(1, "KEYWORD DECOMPOSITION (DecomposerAgent)")
    t0 = time.time()
    state = {"topic": topic, "current_stage": "init"}
    state = await decomposer_agent.run(state)
    keywords = state.get("keywords", [])
    dt = time.time() - t0

    print(f"⏱️ Time taken: {dt:.2f}s")
    print(f"📌 Generated {len(keywords)} domain-grounded search strategies:")
    for i, kw in enumerate(keywords, 1):
        print(f"   {i}. {kw}")

    if not keywords:
        print("❌ Phase 1 failed: No keywords generated. Stopping test.")
        return

    # -------------------------------------------------------------------------
    # PHASE 2: ARXIV PAPER SEARCH & FETCH
    # -------------------------------------------------------------------------
    print_banner(2, "ARXIV PAPER SEARCH & FETCH (arXiv Tool)")
    t0 = time.time()
    # Search using the top 2 keywords (limited for fast testing)
    search_keywords = keywords[:2]
    all_raw_papers: List[PaperMetadata] = []

    for i, kw in enumerate(search_keywords, 1):
        print(f"\n🔍 Search Strategy [{i}/{len(search_keywords)}]: '{kw}'")
        try:
            results = await arxiv_tool.search(kw, topic, max_results=3)
            print(f"   Found {len(results)} papers:")
            for p in results:
                print(f"     • [{p.arxiv_id}] {p.title[:65]}... ({p.published_date})")
            all_raw_papers.extend(results)
        except Exception as e:
            print(f"   ❌ Search failed for '{kw}': {e}")

    # Deduplicate
    seen_ids = set()
    deduped_papers: List[PaperMetadata] = []
    for p in all_raw_papers:
        if p.arxiv_id not in seen_ids:
            seen_ids.add(p.arxiv_id)
            deduped_papers.append(p)

    dt = time.time() - t0
    print(f"\n⏱️ Time taken: {dt:.2f}s")
    print(f"📌 Total papers fetched: {len(all_raw_papers)} → Unique after dedup: {len(deduped_papers)}")

    if not deduped_papers:
        print("❌ Phase 2 failed: No papers fetched from arXiv. Stopping test.")
        return

    # -------------------------------------------------------------------------
    # PHASE 3: RELEVANCE FILTERING
    # -------------------------------------------------------------------------
    print_banner(3, "RELEVANCE FILTERING (RelevanceFilterAgent)")
    t0 = time.time()
    print(f"Evaluating {len(deduped_papers)} papers against topic '{topic}'...")
    
    relevant_papers = await relevance_filter_agent.filter(deduped_papers, topic)
    dt = time.time() - t0

    print(f"\n⏱️ Time taken: {dt:.2f}s")
    print(f"📌 Relevance Filter Results: {len(relevant_papers)} accepted / {len(deduped_papers) - len(relevant_papers)} discarded")
    print("   Accepted Papers:")
    for p in relevant_papers:
        print(f"     ✅ [{p.arxiv_id}] {p.title}")

    if not relevant_papers:
        print("⚠️ No papers passed relevance filter. Taking top raw paper for remaining test phases.")
        target_paper = deduped_papers[0]
    else:
        target_paper = relevant_papers[0]

    # -------------------------------------------------------------------------
    # PHASE 4: PDF TEXT EXTRACTION
    # -------------------------------------------------------------------------
    print_banner(4, f"PDF TEXT EXTRACTION (Paper: {target_paper.arxiv_id})")
    t0 = time.time()
    print(f"Downloading and parsing PDF for: [{target_paper.arxiv_id}] {target_paper.title}")

    state_input = {"paper": target_paper, "topic": topic}
    paper_output = await pdf_extractor_node(state_input)
    dt = time.time() - t0

    print(f"\n⏱️ Time taken: {dt:.2f}s")
    print(f"📌 Extraction Output Summary:")
    print(f"   • Paper ID: {paper_output.paper_id}")
    print(f"   • PDF Path: {paper_output.local_pdf_path}")
    print(f"   • Full Text Length: {len(paper_output.extracted.full_text)} characters")
    print(f"   • Sections Extracted: {list(paper_output.extracted.sections.keys()) if paper_output.extracted.sections else 'None'}")
    print(f"   • Text Snippet (First 300 chars):\n     \"{paper_output.extracted.full_text[:300]}...\"")

    # -------------------------------------------------------------------------
    # PHASE 5: SUMMARIZATION & CRITIC NOTE GENERATION
    # -------------------------------------------------------------------------
    print_banner(5, "SUMMARIZATION & CRITIC NOTE GENERATION")
    t0 = time.time()
    print("Running Summarizer Agent...")
    summarized_output = await summarizer_agent.run(paper_output)
    
    print("Running Critic Agent...")
    final_output = await critic_agent.run(summarized_output)
    dt = time.time() - t0

    note = final_output.knowledge_note
    print(f"\n⏱️ Time taken: {dt:.2f}s")
    if note:
        print(f"📌 Knowledge Note Created Successfully:")
        print(f"   • One-Sentence Summary: {note.one_sentence_summary}")
        print(f"   • Criticality Score: {note.criticality_score:.2f}")
        print(f"   • Concepts: {note.concepts}")
        if note.structured_data:
            sd = note.structured_data
            print(f"   • Objective: {sd.objective}")
            print(f"   • Key Contributions:")
            for c in sd.key_contributions[:3]:
                print(f"       - {c}")
            print(f"   • Limitations: {sd.limitations[:2]}")
            print(f"   • Benchmarks: {sd.benchmarks[:2]}")
    else:
        print("❌ Knowledge note was not created!")

    # -------------------------------------------------------------------------
    # PHASE 6: LAYERED MEMORY STORAGE
    # -------------------------------------------------------------------------
    print_banner(6, "LAYERED MEMORY STORAGE (Disk + Chroma Vector DB)")
    t0 = time.time()
    try:
        await memory_manager.store_paper(final_output, topic)
        dt = time.time() - t0
        print(f"⏱️ Time taken: {dt:.2f}s")
        print(f"✅ Layered storage complete for [{final_output.paper_id}]")
        print(f"   • Disk Artifacts: papers/{final_output.paper_id}/ (knowledge_note.json, metadata.json, paper.pdf)")
        print(f"   • Chroma Vector Store: Indexed in 'research_notes' collection")
        print(f"   • Research Index: Registered in research_index.json")
    except Exception as e:
        print(f"❌ Storage failed: {e}")

    # -------------------------------------------------------------------------
    # PHASE 7: GRAPH TRIPLET EXTRACTION & NEO4J INSERTION
    # -------------------------------------------------------------------------
    print_banner(7, "GRAPH TRIPLET EXTRACTION & NEO4J INSERTION")
    t0 = time.time()
    if not neo4j_client.is_connected():
        print("⚠️ Neo4j is not connected. Skipping Neo4j insertion test.")
    else:
        try:
            print("Extracting property graph entities and relationships...")
            graph_data = await extractor_agent.extract_graph_elements(final_output)
            print(f"   Extracted {len(graph_data.entities)} Entity Nodes:")
            for e in graph_data.entities[:5]:
                desc = (e.description[:50] + "...") if e.description else "No description"
                print(f"     • ({e.name}:{e.type}) - {desc}")
            
            print(f"   Extracted {len(graph_data.relationships)} Relationships:")
            for r in graph_data.relationships[:5]:
                val = f" ({r.value})" if r.value else ""
                print(f"     • ({r.source}) -[:{r.relation}]-> ({r.target}){val}")

            print("\nWriting to Neo4j Graph Database...")
            neo4j_client.write_extracted_graph(
                final_output.paper_id,
                graph_data.entities,
                graph_data.relationships
            )
            dt = time.time() - t0
            print(f"⏱️ Time taken: {dt:.2f}s")
            print(f"✅ Neo4j property graph updated successfully!")
        except Exception as e:
            print(f"❌ Graph extraction/insertion failed: {e}")

    # -------------------------------------------------------------------------
    # PIPELINE TEST SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  ✅ INGESTION PIPELINE PHASE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    test_topic = "Code generation and software engineering agents"
    if len(sys.argv) > 1:
        test_topic = " ".join(sys.argv[1:])

    asyncio.run(test_pipeline_phases(test_topic))

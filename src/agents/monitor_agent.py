"""
Continuous Monitor Agent
========================
Background agent that periodically checks arXiv for new papers
on existing research topics and auto-ingests them.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger

from src.tools.arxiv_tool import arxiv_tool
from src.tools.research_index import research_index
from src.agents.session_manager import session_manager
from src.graphs.ingestion_graph import ingestion_graph
from src.models.schemas import ResearchState, PaperMetadata
from src.agents.memory_manager import memory_manager
from src.agents.pdf_extractor import pdf_extractor_node
from src.agents.summarizer import summarizer_agent
from src.agents.critic_note import critic_agent


class ContinuousMonitorAgent:
    """
    Monitors existing topics for new papers and ingests only the new ones.
    """

    def __init__(self, max_new_papers_per_topic: int = 5, lookback_days: int = 30):
        self.max_new_papers = max_new_papers_per_topic
        self.lookback_days = lookback_days
        self.index = research_index

    async def check_topic_for_new_papers(self, topic: str) -> List[PaperMetadata]:
        """
        Search arXiv for recent papers on the topic and return only the ones
        we have never seen before.
        """
        logger.info(f"Checking for new papers on topic: {topic}")

        # Use a clean, focused query (we can improve this later with the decomposer)
        query = topic
        try:
            candidates = await arxiv_tool.search(
                query=query,
                topic=topic,
                max_results=15   # Fetch a few extra so we can filter
            )
        except Exception as e:
            logger.error(f"arXiv search failed for '{topic}': {e}")
            return []

        # Filter: only keep papers we have never processed
        known_ids = self.index.get_known_paper_ids()
        new_papers = []

        for paper in candidates:
            if paper.arxiv_id not in known_ids:
                new_papers.append(paper)

        # Limit how many we process in one run
        new_papers = new_papers[:self.max_new_papers]

        logger.info(f"Found {len(new_papers)} new papers for topic '{topic}'")
        return new_papers

    async def process_single_paper(self, paper: PaperMetadata, topic: str) -> bool:
        """Run the full per-paper pipeline for one new paper."""
        try:
            logger.info(f"Processing new paper: {paper.arxiv_id} - {paper.title[:60]}...")

            # Reuse the same pipeline nodes
            input_data = {"paper": paper, "topic": topic}

            output = await pdf_extractor_node(input_data)
            output = await summarizer_agent.run(output)
            output = await critic_agent.run(output)

            # Store everything
            await memory_manager.store_paper(output, topic)

            # Register in the Research Index
            self.index.register_paper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                topic=topic
            )

            logger.success(f"Successfully ingested new paper: {paper.arxiv_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to process paper {paper.arxiv_id}: {e}")
            return False

    async def monitor_topic(self, topic: str) -> Dict[str, Any]:
        """Full monitor cycle for one topic."""
        new_papers = await self.check_topic_for_new_papers(topic)

        results = {
            "topic": topic,
            "new_papers_found": len(new_papers),
            "successfully_ingested": 0,
            "failed": 0,
            "paper_ids": []
        }

        for paper in new_papers:
            success = await self.process_single_paper(paper, topic)
            if success:
                results["successfully_ingested"] += 1
                results["paper_ids"].append(paper.arxiv_id)
            else:
                results["failed"] += 1

            # Be polite between papers
            await asyncio.sleep(3)

        # Mark that we checked this topic
        self.index.mark_topic_monitored(topic)

        return results

    async def run_once(self, topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Run one full monitoring cycle over the given topics
        (or all known topics if none provided).
        """
        if topics is None:
            topics = self.index.get_all_topics()

            # Also pull topics from active sessions
            for session in session_manager.list_sessions():
                if session.topic.lower() not in [t.lower() for t in topics]:
                    topics.append(session.topic)

        if not topics:
            logger.warning("No topics found to monitor. Create some sessions first.")
            return []

        logger.info(f"Starting Continuous Monitor for {len(topics)} topics...")

        all_results = []
        for i, topic in enumerate(topics):
            logger.info(f"[{i+1}/{len(topics)}] Monitoring: {topic}")
            result = await self.monitor_topic(topic)
            all_results.append(result)

            # Extra delay between topics
            if i < len(topics) - 1:
                await asyncio.sleep(8)

        # Summary
        total_new = sum(r["successfully_ingested"] for r in all_results)
        logger.success(f"Monitor cycle complete. Ingested {total_new} new papers.")
        return all_results


# Global instance
monitor_agent = ContinuousMonitorAgent()


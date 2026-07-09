"""
Artifact Store - Permanent file-based source of truth
"""

import json
from pathlib import Path
from typing import Dict, Any
from loguru import logger

from src.config import settings
from src.models.schemas import PerPaperOutput


class ArtifactStore:
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or settings.base_dir / "papers"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_paper_artifacts(self, output: PerPaperOutput, topic: str):
        """Save all artifacts for a paper"""
        paper_dir = self.base_dir / output.paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Save PDF (already downloaded by pdf_tools)
        # We just ensure metadata exists

        # metadata.json
        metadata = {
            "arxiv_id": output.paper_id,
            "title": output.metadata.title,
            "authors": [a.model_dump() for a in output.metadata.authors],
            "published_date": str(output.metadata.published_date),
            "topic": topic,
        }
        (paper_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # extracted.json
        if output.extracted:
            (paper_dir / "extracted.json").write_text(
                json.dumps(output.extracted.model_dump(), indent=2, default=str)
            )

        # summary.json
        if output.summary:
            (paper_dir / "summary.json").write_text(
                json.dumps(output.summary.model_dump(), indent=2, default=str)
            )

        # knowledge_note.json
        if output.knowledge_note:
            (paper_dir / "knowledge_note.json").write_text(
                json.dumps(output.knowledge_note.model_dump(), indent=2, default=str)
            )

        logger.success(f"Artifacts saved for {output.paper_id}")


artifact_store = ArtifactStore()
"""
Chroma Vector DB Client - Simple & Production Ready
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings


class ChromaClient:
    def __init__(self, collection_name: str = "research_notes"):
        self.persist_dir = settings.chroma_persist_dir
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Chroma collection '{collection_name}' ready")

    def add_knowledge_note(
        self,
        note_id: str,
        document: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ):
        """Add or update a Knowledge Note"""
        try:
            self.collection.upsert(
                ids=[note_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding] if embedding else None
            )
            logger.success(f"Stored note in Chroma: {note_id}")
        except Exception as e:
            logger.error(f"Failed to store note {note_id} in Chroma: {e}")
            raise

    def query(self, query_text: str, n_results: int = 5, where: Optional[Dict] = None):
        """Semantic search"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        return results

    def get_collection_stats(self):
        return {
            "count": self.collection.count(),
            "name": self.collection.name
        }


# Global instance
chroma_client = ChromaClient()
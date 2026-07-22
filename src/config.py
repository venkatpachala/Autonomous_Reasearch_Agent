"""
Central configuration using Pydantic Settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file into os.environ
dotenv.load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        
    )

    # LLM
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    default_model: str = Field(default="qwen2.5:7b", alias="DEFAULT_MODEL")
    extraction_model: str = Field(default="qwen2.5:7b", alias="EXTRACTION_MODEL")
    critic_model: str = Field(default="qwen2.5:7b", alias="CRITIC_MODEL")

    # LlamaParse
    llamaparse_api_key: Optional[str] = Field(default=None, alias="LLAMAPARSE_API_KEY")

    # Paths
    base_dir: Path = Path(__file__).parent.parent.resolve()
    papers_dir: Path = base_dir / "papers"
    outputs_dir: Path = base_dir / "outputs"
    # Chroma
    chroma_persist_dir: Path = Field(default=Path("./chroma_db"), alias="CHROMA_PERSIST_DIR")
    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")

    def model_post_init(self, __context: Any) -> None:
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Fallback to check standard alternative LlamaParse key name
        if not self.llamaparse_api_key:
            import os
            self.llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")


# Force reload
settings = Settings(_env_file=".env")
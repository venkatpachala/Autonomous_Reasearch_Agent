import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from src.config import settings

class GatewayCache:
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file or (settings.outputs_dir / "gateway_cache.json")
        self.data: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load gateway cache: {e}")
        return {}

    def _save(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(
                json.dumps(self.data, indent=2, default=str),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Failed to save gateway cache: {e}")

    def _make_key(self, model: str, messages: list, temperature: float, schema_name: Optional[str] = None) -> str:
        # Create a stable string representation
        content_parts = []
        for msg in messages:
            content_parts.append(f"{msg.get('role')}:{msg.get('content')}")
        
        raw_key = f"{model}|{'-'.join(content_parts)}|{temperature}|{schema_name or 'none'}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    def get(self, model: str, messages: list, temperature: float, schema_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response if present."""
        key = self._make_key(model, messages, temperature, schema_name)
        if key in self.data:
            logger.info(f"Cache HIT for query key: {key}")
            return self.data[key]
        return None

    def set(self, model: str, messages: list, temperature: float, schema_name: Optional[str], response_data: Dict[str, Any]):
        """Store a response in the cache."""
        key = self._make_key(model, messages, temperature, schema_name)
        self.data[key] = response_data
        self._save()

gateway_cache = GatewayCache()

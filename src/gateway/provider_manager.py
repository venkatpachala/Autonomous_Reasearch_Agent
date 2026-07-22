import httpx
import os
import time
from typing import Dict, Any, List, Optional
from loguru import logger
from src.config import settings

class OllamaProvider:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.ollama_base_url
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    async def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call Ollama chat endpoint synchronously or asynchronously."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
            },
            "stream": False
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        start_time = time.time()
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        latency = time.time() - start_time

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        
        # Simple fallback token estimation if Ollama doesn't return count
        if prompt_tokens == 0:
            prompt_tokens = sum(len(m.get("content", "")).split() for m in messages)
        if completion_tokens == 0:
            completion_tokens = len(data.get("message", {}).get("content", "").split())

        return {
            "text": data.get("message", {}).get("content", ""),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency": latency,
            "provider": "Ollama",
            "model": model,
            "metadata": {"raw_response": data}
        }

    async def embed(self, model: str, text: str) -> List[float]:
        """Generate embedding vector using Ollama's embed endpoint."""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model,
            "prompt": text
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        
        # Handle different response key shapes in older vs newer Ollama versions
        embedding = data.get("embedding")
        if not embedding and "embeddings" in data:
            embedding = data["embeddings"][0]
        return embedding

class OpenAIProvider:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"

    async def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call OpenAI chat completion endpoint using raw HTTP."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment or config.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        start_time = time.time()
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        latency = time.time() - start_time

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        choices = data.get("choices", [])
        text = choices[0].get("message", {}).get("content", "") if choices else ""

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency": latency,
            "provider": "OpenAI",
            "model": model,
            "metadata": {"raw_response": data}
        }

    async def embed(self, model: str, text: str, dimensions: Optional[int] = None) -> List[float]:
        """Generate embedding vector using OpenAI embeddings API.
        
        Args:
            model: Embedding model name (e.g. 'text-embedding-3-small')
            text: Text to embed
            dimensions: Optional output dimension for models that support truncation
                        (text-embedding-3-small supports any dim ≤ 1536).
                        Must match Pinecone index dimension exactly.
        """
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "input": text
        }
        # Only text-embedding-3-* models support the dimensions parameter
        if dimensions and "embedding-3" in model:
            payload["dimensions"] = dimensions

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        
        return data["data"][0]["embedding"]


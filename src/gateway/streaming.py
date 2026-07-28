"""
Token streaming helpers for chat UX.
Uses OpenAI stream when available; otherwise yields one-shot text.
"""

from typing import AsyncIterator, List, Dict, Optional
from loguru import logger
from openai import AsyncOpenAI
from src.config import settings


async def stream_chat_tokens(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> AsyncIterator[str]:
    """
    Yield text deltas as they arrive.
    Falls back to a single full response if streaming is unavailable.
    """
    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # No OpenAI — single shot via gateway
        from src.gateway import gateway

        resp = await gateway.generate(
            task="research_answer",
            messages=messages,
            temperature=temperature,
        )
        text = getattr(resp, "text", None) or ""
        if text:
            yield text
        return

    try:

        client = AsyncOpenAI(api_key=api_key)
        chosen = model or getattr(settings, "default_chat_model", None) or "gpt-4o-mini"

        stream = await client.chat.completions.create(
            model=chosen,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        async for event in stream:
            try:
                delta = event.choices[0].delta.content
            except (AttributeError, IndexError):
                delta = None
            if delta:
                yield delta
    except Exception as e:
        logger.warning(f"Streaming failed ({e}); falling back to non-stream")
        from src.gateway import gateway

        resp = await gateway.generate(
            task="research_answer",
            messages=messages,
            temperature=temperature,
        )
        text = getattr(resp, "text", None) or ""
        if text:
            yield text
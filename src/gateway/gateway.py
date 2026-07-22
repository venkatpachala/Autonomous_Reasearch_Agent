import re
import json
import time
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, ValidationError
from loguru import logger

from src.gateway.request import GatewayRequest
from src.gateway.response import GatewayResponse
from src.gateway.router import router
from src.gateway.model_registry import get_pricing
from src.gateway.provider_manager import OllamaProvider, OpenAIProvider
from src.gateway.cache import gateway_cache
from src.gateway.retries import retry_with_backoff
from src.gateway.circuit_breaker import circuit_breaker
from src.gateway.cost_tracker import cost_tracker
from src.gateway.embeddings import embeddings_gateway

def clean_json_text(text: str) -> str:
    """Strip markdown code block ticks if present."""
    cleaned = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    return cleaned

class AIGateway:
    """
    Main AI Gateway orchestrator managing caching, routing, retries,
    circuit breakers, structured output validation, and cost tracking.
    """
    def __init__(self):
        self.ollama = OllamaProvider()
        self.openai = OpenAIProvider()
        self.embed = embeddings_gateway.embed

    async def generate(
        self,
        task: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        schema_model: Optional[Type[BaseModel]] = None,
        max_tokens: Optional[int] = None,
        retries: int = 3
    ) -> GatewayResponse:
        """
        Main entrypoint for chat generation requests.
        """
        # 1. Resolve routing
        provider, model, fallback = router.route(task)

        # 2. Check Cache
        schema_name = schema_model.__name__ if schema_model else None
        cached_data = gateway_cache.get(model, messages, temperature, schema_name)
        if cached_data:
            # Parse structured output if needed
            structured_obj = None
            if schema_model and cached_data.get("text"):
                try:
                    cleaned = clean_json_text(cached_data["text"])
                    structured_obj = schema_model.model_validate_json(cleaned)
                except Exception as e:
                    logger.warning(f"Failed to rebuild cached structured model: {e}")
            
            return GatewayResponse(
                text=cached_data["text"],
                structured=structured_obj,
                prompt_tokens=cached_data.get("prompt_tokens", 0),
                completion_tokens=cached_data.get("completion_tokens", 0),
                total_tokens=cached_data.get("total_tokens", 0),
                cost=cached_data.get("cost", 0.0),
                latency=0.0,
                model=model,
                provider=provider,
                cached=True
            )

        # 3. Define the actual LLM call executing with retries
        # Copy messages to allow mutation during retry loops (self-correction)
        execution_messages = list(messages)

        async def _call_llm():
            # Check Circuit Breaker
            if not circuit_breaker.can_execute(provider):
                raise RuntimeError(f"Circuit breaker is OPEN for provider '{provider}'")

            try:
                if provider.lower() == "openai":
                    response = await self.openai.generate(model, execution_messages, temperature, max_tokens)
                else:
                    response = await self.ollama.generate(model, execution_messages, temperature, max_tokens)
                
                circuit_breaker.record_success(provider)
            except Exception as e:
                circuit_breaker.record_failure(provider)
                raise e

            text = response["text"]
            structured_output = None

            # 4. Structured Output Validation
            if schema_model:
                try:
                    cleaned_text = clean_json_text(text)
                    structured_output = schema_model.model_validate_json(cleaned_text)
                except (ValidationError, json.JSONDecodeError) as val_err:
                    logger.warning(f"Structured output validation failed: {val_err}")
                    # Feed the error back to the model for self-correction in next attempt
                    execution_messages.append({"role": "assistant", "content": text})
                    execution_messages.append({
                        "role": "user",
                        "content": f"Your response did not conform to the required JSON schema. Error: {val_err}. "
                                   f"Please correct your response and return ONLY valid JSON matching the schema."
                    })
                    raise ValueError(f"JSON schema validation failed: {val_err}")

            response["structured"] = structured_output
            return response

        # 5. Execute with retry backoff
        try:
            res_data = await retry_with_backoff(_call_llm, retries=retries)
        except Exception as primary_err:
            logger.error(f"Primary provider failed: {primary_err}")
            # Try fallback model if configured
            if fallback:
                logger.info(f"Primary path failed. Routing to fallback model '{fallback}' (Ollama).")
                # Override provider/model to fallback (Ollama)
                provider, model = "Ollama", fallback
                try:
                    res_data = await retry_with_backoff(_call_llm, retries=2)
                except Exception as fallback_err:
                    logger.critical(f"Fallback path failed: {fallback_err}")
                    raise fallback_err
            else:
                raise primary_err

        # 6. Calculate cost
        pricing = get_pricing(model)
        cost = (
            (res_data["prompt_tokens"] * pricing.input_cost_per_1m) +
            (res_data["completion_tokens"] * pricing.output_cost_per_1m)
        ) / 1_000_000.0

        # 7. Track costs
        cost_tracker.track_request(
            task=task,
            model=model,
            prompt_tokens=res_data["prompt_tokens"],
            completion_tokens=res_data["completion_tokens"],
            cost=cost,
            latency=res_data["latency"]
        )

        # 8. Cache response
        gateway_cache.set(
            model=model,
            messages=messages,
            temperature=temperature,
            schema_name=schema_name,
            response_data={
                "text": res_data["text"],
                "prompt_tokens": res_data["prompt_tokens"],
                "completion_tokens": res_data["completion_tokens"],
                "total_tokens": res_data["total_tokens"],
                "cost": cost
            }
        )

        return GatewayResponse(
            text=res_data["text"],
            structured=res_data["structured"],
            prompt_tokens=res_data["prompt_tokens"],
            completion_tokens=res_data["completion_tokens"],
            total_tokens=res_data["total_tokens"],
            cost=cost,
            latency=res_data["latency"],
            model=model,
            provider=provider,
            cached=False,
            metadata=res_data.get("metadata", {})
        )

gateway = AIGateway()

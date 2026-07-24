import re
import json
import time
from typing import Dict, Any, List, Optional, Type, Tuple
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
from src.config import settings


def clean_json_text(text: str) -> str:
    """Strip markdown fences and illegal control characters."""
    if not text:
        return ""

    cleaned = text.strip()

    # Remove ```json ... ``` fences
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()

    # Remove illegal C0 control chars (keep \n \r \t)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)

    return cleaned


def extract_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object from text.
    Helps when the model adds prose before/after the JSON.
    """
    cleaned = clean_json_text(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


class GroundednessError(ValueError):
    """Raised when factual consistency check fails after retries."""

    def __init__(self, message: str, last_answer: str):
        super().__init__(message)
        self.last_answer = last_answer


class AIGateway:
    """
    Main AI Gateway: caching, routing, retries, circuit breakers,
    structured output validation, groundedness checks, cost tracking.
    """

    def __init__(self):
        self.ollama = OllamaProvider()
        self.openai = OpenAIProvider()
        self.embed = embeddings_gateway.embed

    # ------------------------------------------------------------------ #
    # Groundedness
    # ------------------------------------------------------------------ #
    async def _verify_groundedness(
        self, answer: str, context: str, model: str) -> Tuple[bool, str]:
        """
        Fast LLM-as-a-judge consistency validation.
        Fail-open on empty/invalid judge responses.
        """
        # Skip if no real context to check against
        if not (context or "").strip() or not (answer or "").strip():
            return True, "Skipped: empty context or answer"

        # Fast path: answer that admits missing info is grounded
        lower = answer.lower()
        insufficient_phrases = [
            "does not contain",
            "not present in the context",
            "context does not include",
            "no information about",
            "cannot determine from the provided",
            "not mentioned in the retrieved",
            "not available in the context",
            "context is insufficient",
            "not stored",
            "not available",]
        if any(p in lower for p in insufficient_phrases):
            return True, "Answer correctly reports insufficient context"

    # Keep judge prompt short — long context causes empty local-model replies
        ctx = context.strip()
        if len(ctx) > 3500:
            ctx = ctx[:3500] + "\n...[truncated]"

        system_prompt = (
            "You are a factual consistency judge.\n"
            "Return ONLY one JSON object, no markdown:\n"
            '{"is_grounded": true, "reason": "short reason"}\n\n'
            "Rules:\n"
            "- is_grounded=false ONLY if the Answer invents specific facts/numbers "
            "not in the Context.\n"
            "- is_grounded=true if the Answer is supported OR correctly says "
            "information is missing.\n")

        human_prompt = f"Context:\n{ctx}\n\nAnswer:\n{answer}\n\nJSON only:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},]

        try:
            local_model = settings.default_model or "qwen2.5:7b"
            response = await self.ollama.generate(
                model=local_model,
                messages=messages,
                temperature=0.0,)
            raw = (response.get("text") if isinstance(response, dict) else None) or ""
            raw = raw.strip()
            if not raw:
                logger.warning(
                "Groundedness checker returned empty — defaulting to True"
                )
                return True, "empty_judge_response"

            # Prefer extract_json_object if available; else first {...} span
            try:
                text = extract_json_object(raw)
            except Exception:
                start, end = raw.find("{"), raw.rfind("}")
                text = raw[start : end + 1] if start != -1 and end > start else raw

            data = json.loads(text)
            is_grounded = bool(data.get("is_grounded", True))
            reason = str(data.get("reason", ""))[:200]
            return is_grounded, reason

        except Exception as e:
            logger.warning(f"Groundedness checker failed: {e}. Defaulting to True.")
            return True, f"judge_error: {e}"

    # ------------------------------------------------------------------ #
    # Structured parse helper
    # ------------------------------------------------------------------ #
    def _parse_structured(
        self, text: str, schema_model: Type[BaseModel]
    ) -> BaseModel:
        """
        Parse model text into a validated Pydantic object.
        Never confuses schema definition with model output.
        """
        candidate = extract_json_object(text)

        # First try strict JSON → model
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from model: {e}") from e

        # Guard: reject accidental schema objects
        if isinstance(data, dict) and ("$defs" in data or data.get("type") == "object"):
            # Looks like a JSON Schema, not an instance
            if "properties" in data or "$schema" in data:
                raise ValueError(
                    "Model returned a JSON Schema definition instead of an instance. "
                    "Expected a filled object matching the schema."
                )

        try:
            return schema_model.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Schema validation failed: {e}") from e

    # ------------------------------------------------------------------ #
    # Main generate
    # ------------------------------------------------------------------ #
    async def generate(
        self,
        task: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        schema_model: Optional[Type[BaseModel]] = None,
        max_tokens: Optional[int] = None,
        retries: int = 3,
    ) -> GatewayResponse:
        provider, model, fallback = router.route(task)

        # Cache
        schema_name = schema_model.__name__ if schema_model else None
        cached_data = gateway_cache.get(model, messages, temperature, schema_name)
        if cached_data:
            structured_obj = None
            if schema_model and cached_data.get("text"):
                try:
                    structured_obj = self._parse_structured(
                        cached_data["text"], schema_model
                    )
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
                cached=True,
            )

        # Working copy of messages
        execution_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        # Inject schema instruction once (not on every retry as a new system msg)
        if schema_model:
            schema_json = json.dumps(schema_model.model_json_schema(), indent=2)
            schema_instruction = (
                "\n\nYou must respond ONLY with a single JSON object that is an "
                "INSTANCE of this schema (fill in the fields). "
                "Do NOT return the schema definition itself.\n"
                f"{schema_json}\n"
                "Return compact valid JSON only. Escape newlines inside strings as \\n."
            )
            system_msg = next(
                (m for m in execution_messages if m["role"] == "system"), None
            )
            if system_msg:
                system_msg["content"] += schema_instruction
            else:
                execution_messages.insert(
                    0, {"role": "system", "content": schema_instruction}
                )

        async def _call_llm():
            if not circuit_breaker.can_execute(provider):
                raise RuntimeError(f"Circuit breaker is OPEN for provider '{provider}'")

            try:
                if provider.lower() == "openai":
                    response = await self.openai.generate(
                        model, execution_messages, temperature, max_tokens
                    )
                else:
                    response = await self.ollama.generate(
                        model, execution_messages, temperature, max_tokens
                    )
                circuit_breaker.record_success(provider)
            except Exception as e:
                circuit_breaker.record_failure(provider)
                raise e

            text = response["text"]
            structured_output = None

            # Structured validation
            if schema_model:
                try:
                    structured_output = self._parse_structured(text, schema_model)
                except Exception as val_err:
                    logger.warning(f"Structured output validation failed: {val_err}")
                    # Self-correction for next retry — still ask for INSTANCE, not schema
                    execution_messages.append({"role": "assistant", "content": text})
                    execution_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous response was invalid: {val_err}\n\n"
                                "Return ONLY a filled JSON object matching the required schema. "
                                "Do NOT return the schema definition. "
                                "Escape all newlines inside string values as \\n. "
                                "No markdown, no prose — JSON only."
                            ),
                        }
                    )
                    raise ValueError(f"JSON schema validation failed: {val_err}")

            # Groundedness (research + synthesis)
            if task in ("research_answer", "synthesis"):
                user_msg = next(
                    (m["content"] for m in execution_messages if m["role"] == "user"),
                    "",
                )
                is_grounded, reason = await self._verify_groundedness(
                    text, user_msg, model
                )
                if not is_grounded:
                    logger.warning(f"Hallucination detected: {reason}")
                    execution_messages.append({"role": "assistant", "content": text})
                    execution_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your response was rejected for ungrounded claims: {reason}\n"
                                "Rewrite so EVERY statement is backed by the provided context. "
                                "If information is missing, say so explicitly."
                            ),
                        }
                    )
                    raise GroundednessError(
                        f"Factual consistency check failed: {reason}",
                        last_answer=text,
                    )

            response["structured"] = structured_output
            return response

        # Execute with retries
        try:
            res_data = await retry_with_backoff(_call_llm, retries=retries)
        except Exception as primary_err:
            logger.error(f"Primary provider failed: {primary_err}")
            if fallback:
                logger.info(
                    f"Routing to fallback model '{fallback}' (Ollama)."
                )
                provider, model = "Ollama", fallback
                try:
                    res_data = await retry_with_backoff(_call_llm, retries=2)
                except Exception as fallback_err:
                    logger.critical(f"Fallback path failed: {fallback_err}")
                    raise fallback_err
            else:
                raise primary_err

        # Cost
        pricing = get_pricing(model)
        cost = (
            (res_data["prompt_tokens"] * pricing.input_cost_per_1m)
            + (res_data["completion_tokens"] * pricing.output_cost_per_1m)
        ) / 1_000_000.0

        cost_tracker.track_request(
            task=task,
            model=model,
            prompt_tokens=res_data["prompt_tokens"],
            completion_tokens=res_data["completion_tokens"],
            cost=cost,
            latency=res_data["latency"],
        )

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
                "cost": cost,
            },
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
            metadata=res_data.get("metadata", {}),
        )


gateway = AIGateway()
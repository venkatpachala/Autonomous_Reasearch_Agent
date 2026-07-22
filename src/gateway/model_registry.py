from typing import Dict, Any, List
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    provider: str
    model_name: str
    fallback: Optional[str] = None  # Task-specific fallback model if primary fails

class ModelPricing(BaseModel):
    input_cost_per_1m: float = 0.0  # USD per million tokens
    output_cost_per_1m: float = 0.0  # USD per million tokens

from typing import Optional

# Mapping of task types to primary and fallback model specifications
TASK_MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "keyword_generation": ModelConfig(
        provider="Ollama",
        model_name="qwen2.5:7b"
    ),
    "summary": ModelConfig(
        provider="Ollama",
        model_name="qwen2.5:7b"
    ),
    "research_answer": ModelConfig(
        provider="OpenAI",
        model_name="gpt-4o-mini",
        fallback="qwen2.5:7b"  # Falls back to Ollama locally
    ),
    "evaluation": ModelConfig(
        provider="Ollama",
        model_name="qwen2.5:7b"
    ),
    "default": ModelConfig(
        provider="Ollama",
        model_name="qwen2.5:7b"
    )
}

# Pricing directory (costs in USD per 1,000,000 tokens)
MODEL_PRICING_REGISTRY: Dict[str, ModelPricing] = {
    # OpenAI models
    "gpt-4o-mini": ModelPricing(input_cost_per_1m=0.15, output_cost_per_1m=0.60),
    "gpt-4o": ModelPricing(input_cost_per_1m=2.50, output_cost_per_1m=10.00),
    "text-embedding-3-small": ModelPricing(input_cost_per_1m=0.02, output_cost_per_1m=0.00),
    
    # Local models (Ollama is running locally, so cost is $0)
    "qwen2.5:7b": ModelPricing(input_cost_per_1m=0.0, output_cost_per_1m=0.0),
    "nomic-embed-text": ModelPricing(input_cost_per_1m=0.0, output_cost_per_1m=0.0),
    "default": ModelPricing(input_cost_per_1m=0.0, output_cost_per_1m=0.0)
}

def get_pricing(model_name: str) -> ModelPricing:
    """Get the pricing configuration for a given model, defaulting to zero costs."""
    return MODEL_PRICING_REGISTRY.get(model_name, MODEL_PRICING_REGISTRY["default"])

def get_task_config(task: str) -> ModelConfig:
    """Get the model configuration for a task, defaulting to the generic config."""
    return TASK_MODEL_REGISTRY.get(task, TASK_MODEL_REGISTRY["default"])

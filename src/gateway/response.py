from typing import Optional, Any, Dict
from pydantic import BaseModel, Field

class GatewayResponse(BaseModel):
    """
    Standard response format returned by the AI Gateway.
    """
    text: str = Field(..., description="The raw text response from the model")
    structured: Optional[Any] = Field(None, description="The validated, parsed Pydantic schema model instance if requested")
    prompt_tokens: int = Field(0, description="Number of tokens in the input prompt")
    completion_tokens: int = Field(0, description="Number of tokens in the generated response")
    total_tokens: int = Field(0, description="Total tokens consumed (prompt + completion)")
    cost: float = Field(0.0, description="Estimated USD cost of the request")
    latency: float = Field(0.0, description="Total time taken for request completion in seconds")
    model: str = Field(..., description="The specific model used for this request")
    provider: str = Field(..., description="The model provider (e.g. 'Ollama', 'OpenAI')")
    cached: bool = Field(False, description="True if response was retrieved from the cache")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific response metadata")

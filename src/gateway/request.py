from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field

class GatewayRequest(BaseModel):
    """
    Standard request format for the AI Gateway.
    """
    task: str = Field(..., description="The research or processing task (e.g. 'keyword_generation', 'summary', 'research_answer', 'evaluation')")
    messages: List[Dict[str, str]] = Field(..., description="List of messages in standard chat format: [{'role': 'system'|'user'|'assistant', 'content': '...'}]")
    temperature: float = Field(0.2, description="Sampling temperature for the LLM")
    schema_model: Optional[Type[BaseModel]] = Field(None, description="Optional Pydantic class to validate and structure the response")
    max_tokens: Optional[int] = Field(None, description="Maximum completion tokens")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the backend provider")

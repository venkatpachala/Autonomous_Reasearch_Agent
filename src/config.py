from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    OLLAMA_MODEL: str="qwen2.5:7b"
    TEMPERATURE: float=0.6
    LLAMA_CLOUD_API_KEY: str="llx-2cgyFLturoBCGY2S4tIX4c451hDkPhQt6f92GVITEGZ7Z1BL"

settings=Settings()
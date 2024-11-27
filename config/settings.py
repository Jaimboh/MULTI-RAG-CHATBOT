import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv(dotenv_path="./.env")

def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

class LLMSettings(BaseSettings):
    """Base settings for Language Model configurations."""
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3

class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
class GroqSettings(LLMSettings):
    """Groq-specific settings extending LLMSettings."""
    api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    default_model: str = Field(default="llama-3.1-70b-versatile")

class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER"))
    password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD"))
    host: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST"))
    port: str = Field(default_factory=lambda: os.getenv("POSTGRES_PORT"))
    image_db_name: str = "demonstration-image-database"
    pdf_db_name: str = "demonstration-pdf-database"
    pdf_collection_name: str = "pdf_documents"

class VectorStoreSettings(BaseSettings):
    """Settings for the VectorStore."""
    table_name: str = "image_embeddings"
    embedding_dimensions: int = 1536
    time_partition_interval: timedelta = timedelta(days=7)

class Settings(BaseSettings):
    """Main settings class combining all sub-settings."""
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)

@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings

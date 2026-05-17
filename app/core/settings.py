from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    CHAT_MODEL: str = "openai/gpt-4o-mini"

    EMBED_MODEL: str = "denaya/indoSBERT-large"

    UPLOAD_DIR: Path = Path("./uploads")
    CHROMA_DIR: Path = Path("./chroma")
    SQLITE_URL: str = "sqlite:///./app.db"

    TOP_K: int = 5

    COLLECTION_NAME: str = "indo_legal_chunks"


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

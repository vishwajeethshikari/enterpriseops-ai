
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "EnterpriseOps AI"
    environment: str = os.getenv("APP_ENV", "development")

    java_base_url: str = os.getenv(
        "JAVA_BASE_URL",
        "http://localhost:8080",
    )

    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    ollama_chat_model: str = os.getenv(
        "OLLAMA_CHAT_MODEL",
        "qwen2.5:3b",
    )

    ollama_embedding_model: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        "nomic-embed-text",
    )

    request_timeout_seconds: float = float(
        os.getenv("REQUEST_TIMEOUT_SECONDS", "15")
    )


settings = Settings()

"""Configuration settings using pydantic-settings."""

import os
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Google Cloud
    google_application_credentials: Optional[str] = None

    # Gemini API
    gemini_api_key: str
    gemini_api_keys: Optional[str] = None  # Comma-separated for multiple projects

    # Rate Limiting
    max_concurrent_requests: int = 10
    gemini_rate_limit_per_minute: int = 60
    gemini_max_retries: int = 3
    gemini_retry_backoff_base: float = 2.0

    # App Config
    environment: str = "local"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api"

    @property
    def gemini_api_keys_list(self) -> List[str]:
        """Parse comma-separated API keys into a list."""
        if self.gemini_api_keys:
            return [key.strip() for key in self.gemini_api_keys.split(",") if key.strip()]
        return [self.gemini_api_key] if self.gemini_api_key else []

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_local(self) -> bool:
        """Check if running in local environment."""
        return self.environment.lower() == "local"


# Global settings instance
settings = Settings()


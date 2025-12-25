"""Configuration settings using pydantic-settings."""

import os
from typing import Dict, List, Optional, Tuple

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Gemini Model Rate Limits (RPM: Requests Per Minute, TPM: Tokens Per Minute)
# Source: https://ai.google.dev/pricing
GEMINI_MODEL_LIMITS: Dict[str, Dict[str, int]] = {
    "gemini-2.0-flash": {
        "rpm": 2000,  # Requests per minute
        "tpm": 4_000_000,  # Tokens per minute
        "optimal_concurrency": 50,  # Optimal concurrent requests per API key
    },
    "gemini-2.5-pro": {
        "rpm": 150,  # Requests per minute
        "tpm": 2_000_000,  # Tokens per minute
        "optimal_concurrency": 10,  # Optimal concurrent requests per API key
    },
    "gemini-2.5-flash": {
        "rpm": 1000,  # Requests per minute
        "tpm": 2_000_000,  # Tokens per minute
        "optimal_concurrency": 35,
    },
    "gemini-3-flash": {
        "rpm": 1000,  # Requests per minute
        "tpm": 1_000_000,  # Tokens per minute
        "optimal_concurrency": 50,
    },
}

# Default model to use
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

# Dynamic model selection configuration
DYNAMIC_MODEL_CONFIG = {
    "fast": "gemini-2.0-flash",  # Model to use for large batches (above threshold)
    "slow": "gemini-2.5-pro",  # Model to use for small batches (below threshold)
    "threshold": 200,  # Number of items threshold for dynamic selection
}


def get_model_limits(model_name: str) -> Dict[str, int]:
    """
    Get rate limits for a specific Gemini model.
    
    Args:
        model_name: Name of the Gemini model (e.g., 'gemini-2.0-flash')
        
    Returns:
        Dict with 'rpm', 'tpm', and 'optimal_concurrency'
        
    Raises:
        ValueError: If model is not supported
    """
    if model_name not in GEMINI_MODEL_LIMITS:
        available = ", ".join(GEMINI_MODEL_LIMITS.keys())
        raise ValueError(
            f"Model '{model_name}' not supported. Available models: {available}"
        )
    return GEMINI_MODEL_LIMITS[model_name].copy()


def get_dynamic_model(
    item_count: int,
    threshold: Optional[int] = None,
    fast_model: Optional[str] = None,
    slow_model: Optional[str] = None,
) -> str:
    """
    Get model name based on dynamic selection logic.
    
    Args:
        item_count: Number of items to process
        threshold: Optional threshold override (defaults to DYNAMIC_MODEL_CONFIG["threshold"])
        fast_model: Optional fast model override (defaults to DYNAMIC_MODEL_CONFIG["fast"])
        slow_model: Optional slow model override (defaults to DYNAMIC_MODEL_CONFIG["slow"])
        
    Returns:
        Model name to use
    """
    threshold_value = threshold or DYNAMIC_MODEL_CONFIG["threshold"]
    fast = fast_model or DYNAMIC_MODEL_CONFIG["fast"]
    slow = slow_model or DYNAMIC_MODEL_CONFIG["slow"]
    
    if item_count < threshold_value:
        return slow  # Use slow model (Pro) for small batches
    else:
        return fast  # Use fast model (Flash) for large batches


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
    gemini_api_key: Optional[str] = None
    gemini_api_keys: Optional[str] = None  # Comma-separated for multiple projects
    gemini_model: str = DEFAULT_GEMINI_MODEL  # Model to use (default: flash)

    @model_validator(mode="after")
    def validate_api_keys(self):
        """Validate that at least one API key is configured."""
        if not self.gemini_api_key and not self.gemini_api_keys:
            raise ValueError(
                "Either GEMINI_API_KEY or GEMINI_API_KEYS must be configured"
            )
        return self

    # Retry Config (still configurable)
    gemini_max_retries: int = 3
    gemini_retry_backoff_base: float = 2.0

    # Extractor Config
    extractor_timeout_per_item: int = 120  # Timeout per item in seconds

    # Dynamic Model Config
    dynamic_model_threshold: int = DYNAMIC_MODEL_CONFIG["threshold"]  # Threshold for dynamic model selection
    dynamic_model_fast: str = DYNAMIC_MODEL_CONFIG["fast"]  # Model for large batches (>= threshold)
    dynamic_model_slow: str = DYNAMIC_MODEL_CONFIG["slow"]  # Model for small batches (< threshold)

    # App Config
    environment: str = "local"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api"
    port: int = 8001  # Server port

    @property
    def gemini_api_keys_list(self) -> List[str]:
        """Parse comma-separated API keys into a list."""
        if self.gemini_api_keys:
            return [key.strip() for key in self.gemini_api_keys.split(",") if key.strip()]
        return [self.gemini_api_key] if self.gemini_api_key else []

    @property
    def gemini_model_limits(self) -> Dict[str, int]:
        """Get rate limits for the configured Gemini model."""
        return get_model_limits(self.gemini_model)

    @property
    def max_concurrent_requests(self) -> int:
        """
        Calculate optimal concurrency based on model and number of API keys.
        
        Each API key can handle a certain number of concurrent requests based on the model.
        With multiple keys, we scale linearly.
        """
        limits = self.gemini_model_limits
        num_keys = len(self.gemini_api_keys_list)
        return limits["optimal_concurrency"] * num_keys

    @property
    def gemini_rpm_limit(self) -> int:
        """Get RPM limit for the configured model."""
        return self.gemini_model_limits["rpm"]

    @property
    def gemini_tpm_limit(self) -> int:
        """Get TPM limit for the configured model."""
        return self.gemini_model_limits["tpm"]

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


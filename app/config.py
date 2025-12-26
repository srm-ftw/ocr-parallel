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

    # Full Process Config
    output_bucket: Optional[str] = None  # GCS bucket for output files
    stats_bucket: Optional[str] = None  # GCS bucket for statistics JSON files
    full_process_max_concurrent_escrituras: int = 3  # Max concurrent escrituras processing
    chunk_size: int = 40  # Pages per chunk
    chunk_overlap: int = 5  # Pages of overlap between chunks

    # App Config
    environment: str = "dev"  # dev or prod (prod includes local-tunnel)
    log_level: str = "INFO"
    api_v1_prefix: str = "/api"
    port: int = 8001  # Server port
    api_key: Optional[str] = None  # API key for authentication

    # Uvicorn Config
    uvicorn_workers: int = 1  # Number of worker processes
    uvicorn_timeout_keep_alive: Optional[int] = None  # Keep-alive timeout (None = auto based on environment)
    uvicorn_timeout_graceful_shutdown: Optional[int] = None  # Graceful shutdown timeout (None = auto)
    uvicorn_limit_concurrency: Optional[int] = None  # Max concurrent connections (None = auto)
    uvicorn_backlog: int = 2048  # Backlog size

    # Request Limits
    max_request_size_mb: int = 50  # Maximum request body size in MB

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
        """Check if running in production environment (prod includes local-tunnel)."""
        return self.environment.lower() == "prod"

    @property
    def is_dev(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "dev"

    @property
    def uvicorn_reload(self) -> bool:
        """Whether to enable auto-reload (only in dev)."""
        return self.is_dev

    @property
    def uvicorn_timeout_keep_alive_seconds(self) -> int:
        """Get keep-alive timeout based on environment."""
        if self.uvicorn_timeout_keep_alive is not None:
            return self.uvicorn_timeout_keep_alive
        # Defaults based on environment
        return 5 if self.is_dev else 30

    @property
    def uvicorn_timeout_graceful_shutdown_seconds(self) -> int:
        """Get graceful shutdown timeout based on environment."""
        if self.uvicorn_timeout_graceful_shutdown is not None:
            return self.uvicorn_timeout_graceful_shutdown
        # Defaults based on environment
        return 30 if self.is_dev else 60

    @property
    def uvicorn_limit_concurrency_value(self) -> Optional[int]:
        """Get concurrency limit based on environment."""
        if self.uvicorn_limit_concurrency is not None:
            return self.uvicorn_limit_concurrency
        # Defaults based on environment
        return 50 if self.is_dev else 100

    @property
    def max_request_size_bytes(self) -> int:
        """Get maximum request size in bytes."""
        return self.max_request_size_mb * 1024 * 1024


# Global settings instance
settings = Settings()


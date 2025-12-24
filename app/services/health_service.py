"""Health check service."""

import logging
from typing import Dict, Any

from app.clients.gcs_client import GCSClient
from app.clients.gemini_client import GeminiClient
from app import __version__

logger = logging.getLogger(__name__)


class HealthService:
    """Service for health checks."""

    def __init__(self):
        """Initialize health service."""
        self.gcs_client = GCSClient()
        self.gemini_client = GeminiClient()

    async def check_health(self) -> Dict[str, Any]:
        """
        Perform health checks for all dependencies.

        Returns:
            Dict with health status and check results
        """
        checks: Dict[str, str] = {}
        all_healthy = True

        # Check GCS
        try:
            # Try to access the client (lightweight check)
            # If this fails, GCS is not accessible
            _ = self.gcs_client.client
            checks["gcs"] = "ok"
        except Exception as e:
            logger.warning(f"GCS health check failed: {e}")
            checks["gcs"] = f"error: {str(e)}"
            all_healthy = False

        # Check Gemini (just check if client is initialized)
        try:
            if self.gemini_client.api_keys:
                checks["gemini"] = "ok"
            else:
                checks["gemini"] = "error: no API keys configured"
                all_healthy = False
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            checks["gemini"] = f"error: {str(e)}"
            all_healthy = False

        return {
            "status": "healthy" if all_healthy else "degraded",
            "version": __version__,
            "checks": checks,
        }


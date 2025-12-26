#!/usr/bin/env python3
"""Script to run the OCR endpoint locally.

This script automatically configures uvicorn based on the ENVIRONMENT setting:
- dev: Development mode with reload enabled, shorter timeouts
- prod: Production mode (or local with tunnel) with reload disabled, longer timeouts
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from app.config import settings

if __name__ == "__main__":
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("Warning: .env file not found. Make sure to create it with required variables.")
        print("See .env.example for reference.")

    # Determine environment
    env = settings.environment.lower()
    is_dev = settings.is_dev
    is_prod = settings.is_production

    print(f"Starting OCR endpoint service...")
    print(f"  Environment: {env}")
    print(f"  Port: {settings.port}")
    print(f"  Workers: {settings.uvicorn_workers}")
    print(f"  Reload: {settings.uvicorn_reload}")
    print(f"  Timeout keep-alive: {settings.uvicorn_timeout_keep_alive_seconds}s")
    print(f"  Max request size: {settings.max_request_size_mb}MB")
    print(f"  Concurrency limit: {settings.uvicorn_limit_concurrency_value}")

    # Run the server with environment-appropriate settings
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.uvicorn_reload,
        workers=settings.uvicorn_workers if not settings.uvicorn_reload else 1,
        timeout_keep_alive=settings.uvicorn_timeout_keep_alive_seconds,
        timeout_graceful_shutdown=settings.uvicorn_timeout_graceful_shutdown_seconds,
        limit_concurrency=settings.uvicorn_limit_concurrency_value,
        backlog=settings.uvicorn_backlog,
        log_level=settings.log_level.lower(),
    )


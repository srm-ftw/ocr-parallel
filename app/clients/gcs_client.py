"""GCS client for downloading PDFs from Google Cloud Storage."""

import json
import logging
import os
from io import BytesIO
from typing import Any, Dict, Optional

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

from app.config import settings
from app.core.exceptions import GCSException

logger = logging.getLogger(__name__)


class GCSClient:
    """Client for interacting with Google Cloud Storage."""

    def __init__(self):
        """Initialize GCS client."""
        try:
            # Use credentials file if specified in settings
            if settings.google_application_credentials:
                credentials_path = settings.google_application_credentials
                if os.path.exists(credentials_path):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                    logger.info(f"Using GCS credentials from: {credentials_path}")
                else:
                    logger.warning(
                        f"Credentials file not found: {credentials_path}. "
                        "Trying default credentials."
                    )
            elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                logger.info(
                    f"Using GCS credentials from environment: "
                    f"{os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}"
                )

            self.client = storage.Client()
            logger.info("GCS client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise GCSException(f"Failed to initialize GCS client: {e}") from e

    async def download_pdf(self, bucket_name: str, file_name: str) -> bytes:
        """
        Download a PDF file from GCS.

        Args:
            bucket_name: Name of the GCS bucket
            file_name: Name of the file in the bucket

        Returns:
            bytes: The PDF file content as bytes

        Raises:
            GCSException: If the download fails
        """
        try:
            logger.debug(f"Downloading {file_name} from bucket {bucket_name}")
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_name)

            if not blob.exists():
                raise GCSException(f"File {file_name} not found in bucket {bucket_name}")

            # Download as bytes
            pdf_bytes = blob.download_as_bytes()
            logger.debug(f"Successfully downloaded {file_name} ({len(pdf_bytes)} bytes)")
            return pdf_bytes

        except GoogleCloudError as e:
            logger.error(f"GCS error downloading {file_name}: {e}")
            raise GCSException(f"GCS error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error downloading {file_name}: {e}")
            raise GCSException(f"Failed to download file: {e}") from e

    async def file_exists(self, bucket_name: str, file_name: str) -> bool:
        """
        Check if a file exists in GCS.

        Args:
            bucket_name: Name of the GCS bucket
            file_name: Name of the file in the bucket

        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_name)
            return blob.exists()
        except Exception as e:
            logger.warning(f"Error checking file existence: {e}")
            return False

    async def upload_json(
        self, bucket_name: str, file_name: str, data: Dict[str, Any], content_type: str = "application/json"
    ) -> None:
        """
        Upload JSON data to GCS.

        Args:
            bucket_name: Name of the GCS bucket
            file_name: Name of the file in the bucket
            data: Dictionary to serialize as JSON
            content_type: Content type for the blob (default: application/json)

        Raises:
            GCSException: If the upload fails
        """
        try:
            logger.debug(f"Uploading JSON to {file_name} in bucket {bucket_name}")
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_name)

            # Serialize to JSON
            json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

            # Upload with content type
            blob.upload_from_string(json_bytes, content_type=content_type)
            logger.info(f"Successfully uploaded {file_name} ({len(json_bytes)} bytes) to bucket {bucket_name}")
        except GoogleCloudError as e:
            logger.error(f"GCS error uploading {file_name}: {e}")
            raise GCSException(f"GCS error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error uploading {file_name}: {e}")
            raise GCSException(f"Failed to upload file: {e}") from e


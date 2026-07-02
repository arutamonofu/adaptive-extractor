"""Client for MinerU Web API to extract text and figures from PDF files."""

import os
import time
import zipfile
import logging
import requests
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from ae.core.config.optimization_settings import MinerUParserConfig

logger = logging.getLogger(__name__)


class MinerUClient:
    """Client for MinerU Web API to extract text and figures from PDF files."""

    def __init__(self, config: MinerUParserConfig):
        self.config = config
        self.api_token = os.environ.get("MINERU_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "MINERU_API_TOKEN environment variable is not set. "
                "Please add 'MINERU_API_TOKEN=your_token' to your .env file."
            )

    def _get_headers(self) -> Dict[str, str]:
        """Form authorization headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }

    def _retry_request(
        self,
        method: str,
        url: str,
        max_retries: int = 3,
        delay: float = 2.0,
        **kwargs
    ) -> requests.Response:
        """Helper to execute requests with simple exponential backoff retry on network errors or 5xx."""
        last_err = None
        current_delay = delay
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.request(method, url, **kwargs)
                if 500 <= response.status_code < 600:
                    logger.warning(
                        f"Server error {response.status_code} on attempt {attempt}/{max_retries}. "
                        f"Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= 2
                    continue
                return response
            except requests.RequestException as e:
                last_err = e
                logger.warning(
                    f"Network error on attempt {attempt}/{max_retries}: {e}. "
                    f"Retrying in {current_delay}s..."
                )
                time.sleep(current_delay)
                current_delay *= 2
        if last_err:
            raise last_err
        raise RuntimeError("Request failed after retries")

    def request_upload_url(self, file_name: str) -> Tuple[str, List[str]]:
        """Request MinerU upload URL for a file.

        Returns:
            Tuple of (batch_id, list_of_upload_urls)
        """
        url = f"{self.config.api_url}/file-urls/batch"
        payload = {
            "files": [{"name": file_name}],
            "model_version": self.config.model_version,
        }

        response = self._retry_request("POST", url, headers=self._get_headers(), json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != 0:
            raise RuntimeError(f"Failed to request upload URL: {data.get('msg')}")

        batch_id = data["data"]["batch_id"]
        file_urls = data["data"]["file_urls"]
        return batch_id, file_urls

    def upload_file(self, file_path: str, upload_url: str) -> None:
        """Upload local file using PUT to the specified upload URL."""
        with open(file_path, "rb") as f:
            # We do NOT set Content-Type header as required by MinerU API
            response = self._retry_request("PUT", upload_url, data=f, timeout=300)
            if response.status_code not in (200, 201):
                raise RuntimeError(
                    f"Failed to upload file. HTTP {response.status_code}: {response.text}"
                )

    def poll_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Poll job status until 'done' or 'failed' or timeout."""
        url = f"{self.config.api_url}/extract-results/batch/{batch_id}"
        start_time = time.time()

        while time.time() - start_time < self.config.poll_timeout:
            response = self._retry_request("GET", url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                raise RuntimeError(f"Error polling status: {data.get('msg')}")

            extract_results = data["data"].get("extract_result", [])
            if not extract_results:
                time.sleep(self.config.poll_interval)
                continue

            result = extract_results[0]
            state = result.get("state")

            if state == "done":
                return result
            elif state == "failed":
                raise RuntimeError(
                    f"Parse failed: {result.get('err_msg', 'unknown error')}"
                )
            else:
                progress = result.get("extract_progress", {})
                extracted = progress.get("extracted_pages", "?")
                total = progress.get("total_pages", "?")
                logger.info(f"MinerU status: {state} | Pages: {extracted}/{total}")

            time.sleep(self.config.poll_interval)

        raise TimeoutError(f"Timeout waiting for MinerU parsing ({self.config.poll_timeout}s)")

    def download_and_extract_zip(self, zip_url: str, output_dir: str) -> str:
        """Download output ZIP and extract it to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, "result.zip")

        logger.info(f"Downloading ZIP archive from {zip_url}...")
        response = self._retry_request("GET", zip_url, timeout=300, stream=True)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Extracting ZIP archive to {output_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)

        os.remove(zip_path)
        return output_dir

    def parse_pdf(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """Upload, parse, poll and extract PDF results.

        Returns:
            The raw JSON result dictionary returned by MinerU.
        """
        pdf_path = os.path.abspath(pdf_path)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        file_name = os.path.basename(pdf_path)

        logger.info(f"[MinerU] Requesting upload URL for '{file_name}'...")
        batch_id, upload_urls = self.request_upload_url(file_name)
        logger.info(f"[MinerU] batch_id: {batch_id}")

        logger.info("[MinerU] Uploading file...")
        self.upload_file(pdf_path, upload_urls[0])
        logger.info("[MinerU] File uploaded successfully.")

        logger.info("[MinerU] Polling batch status...")
        result = self.poll_batch_status(batch_id)
        zip_url = result.get("full_zip_url")
        if not zip_url:
            raise RuntimeError("MinerU did not return full_zip_url link.")

        logger.info("[MinerU] Downloading and extracting ZIP...")
        self.download_and_extract_zip(zip_url, output_dir)
        logger.info(f"[MinerU] Done. Results extracted to {output_dir}")

        return result

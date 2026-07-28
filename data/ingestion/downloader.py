"""Download DHL public PDF documents listed in data/sources.py."""

import logging
import time
from pathlib import Path

import httpx

from data.sources import SOURCES, DocumentSource

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


def download_source(source: DocumentSource, dest_dir: Path = RAW_DIR) -> Path:
    """Download a single document's PDF, skipping it if already present.

    Retries transient HTTP failures with exponential backoff.

    Args:
        source: The document source to download.
        dest_dir: Directory to save the PDF into.

    Returns:
        Path to the downloaded (or already-existing) PDF file.

    Raises:
        httpx.HTTPError: If the download fails after all retries.
        ValueError: If the response is not a PDF.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{source.doc_id}.pdf"

    if dest_path.exists():
        logger.info("Skipping %s, already downloaded", source.doc_id)
        return dest_path

    last_error: httpx.HTTPError | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = httpx.get(str(source.url), follow_redirects=True, timeout=30.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            last_error = exc
            wait = BACKOFF_SECONDS * (2**attempt)
            logger.warning(
                "Download failed for %s (attempt %d/%d): %s. Retrying in %.1fs",
                source.doc_id,
                attempt + 1,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
            continue

        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower():
            raise ValueError(
                f"Expected PDF for {source.doc_id}, got content-type '{content_type}'"
            )

        dest_path.write_bytes(response.content)
        logger.info("Downloaded %s to %s", source.doc_id, dest_path)
        return dest_path

    raise httpx.HTTPError(
        f"Failed to download {source.doc_id} after {MAX_RETRIES} attempts"
    ) from last_error


def download_all(
    sources: list[DocumentSource] = SOURCES, dest_dir: Path = RAW_DIR
) -> list[Path]:
    """Download every document in sources, skipping ones already present.

    Individual failures are logged and skipped rather than aborting the
    whole run.

    Args:
        sources: Documents to download.
        dest_dir: Directory to save PDFs into.

    Returns:
        Paths to all successfully downloaded (or already-existing) PDFs.
    """
    paths: list[Path] = []
    for source in sources:
        try:
            paths.append(download_source(source, dest_dir))
        except (httpx.HTTPError, ValueError) as exc:
            logger.error("Skipping %s due to error: %s", source.doc_id, exc)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all()

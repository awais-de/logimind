"""Unit tests for data/ingestion/downloader.py."""

from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

from data.ingestion.downloader import download_all, download_source
from data.sources import DocumentSource

SOURCE = DocumentSource(
    doc_id="test-doc",
    name="Test Document",
    url="https://example.com/test-doc.pdf",
    category="test",
)


def _pdf_response(content: bytes = b"%PDF-1.4 fake pdf content") -> Mock:
    response = Mock(spec=httpx.Response)
    response.headers = {"content-type": "application/pdf"}
    response.content = content
    response.raise_for_status = Mock()
    return response


def test_download_source_writes_file(tmp_path: Path) -> None:
    with patch("data.ingestion.downloader.httpx.get", return_value=_pdf_response()) as mock_get:
        result = download_source(SOURCE, dest_dir=tmp_path)

    assert result == tmp_path / "test-doc.pdf"
    assert result.read_bytes() == b"%PDF-1.4 fake pdf content"
    mock_get.assert_called_once()


def test_download_source_skips_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "test-doc.pdf"
    existing.write_bytes(b"already here")

    with patch("data.ingestion.downloader.httpx.get") as mock_get:
        result = download_source(SOURCE, dest_dir=tmp_path)

    assert result == existing
    mock_get.assert_not_called()


def test_download_source_raises_on_non_pdf_content_type(tmp_path: Path) -> None:
    response = Mock(spec=httpx.Response)
    response.headers = {"content-type": "text/html"}
    response.raise_for_status = Mock()

    with patch("data.ingestion.downloader.httpx.get", return_value=response):
        with pytest.raises(ValueError, match="Expected PDF"):
            download_source(SOURCE, dest_dir=tmp_path)


def test_download_source_retries_then_raises(tmp_path: Path) -> None:
    with patch(
        "data.ingestion.downloader.httpx.get",
        side_effect=httpx.ConnectError("boom"),
    ) as mock_get, patch("data.ingestion.downloader.time.sleep"):
        with pytest.raises(httpx.HTTPError, match="Failed to download"):
            download_source(SOURCE, dest_dir=tmp_path)

    assert mock_get.call_count == 3


def test_download_all_skips_failures_and_continues(tmp_path: Path) -> None:
    good_source = DocumentSource(
        doc_id="good-doc",
        name="Good Document",
        url="https://example.com/good-doc.pdf",
        category="test",
    )
    bad_source = DocumentSource(
        doc_id="bad-doc",
        name="Bad Document",
        url="https://example.com/bad-doc.pdf",
        category="test",
    )

    def fake_get(url: str, **kwargs: object) -> Mock:
        if "bad-doc" in url:
            response = Mock(spec=httpx.Response)
            response.headers = {"content-type": "text/html"}
            response.raise_for_status = Mock()
            return response
        return _pdf_response()

    with patch("data.ingestion.downloader.httpx.get", side_effect=fake_get):
        results = download_all([good_source, bad_source], dest_dir=tmp_path)

    assert results == [tmp_path / "good-doc.pdf"]

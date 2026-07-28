"""Unit tests for data/ingestion/embedder.py."""

from datetime import date
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import APIConnectionError

from data.ingestion.chunker import Chunk
from data.ingestion.embedder import EmbeddingClient, embed_chunks

SOURCE_KWARGS = {
    "doc_id": "test-doc",
    "doc_name": "Test Document",
    "category": "test",
    "region": None,
    "publication_date": date(2024, 1, 1),
}


def _chunk(chunk_id: str, page_number: int, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, page_number=page_number, text=text, **SOURCE_KWARGS)


def _embedding_response(vectors: list[list[float]]) -> Mock:
    response = Mock()
    response.data = [Mock(embedding=vector) for vector in vectors]
    return response


def test_embed_batch_returns_vectors_in_order() -> None:
    client = EmbeddingClient(api_key="test-key")
    with patch.object(
        client._client.embeddings, "create", return_value=_embedding_response([[0.1, 0.2], [0.3, 0.4]])
    ) as mock_create:
        vectors = client.embed_batch(["text one", "text two"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    mock_create.assert_called_once()


def test_embed_batch_retries_then_succeeds() -> None:
    client = EmbeddingClient(api_key="test-key")
    error = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
    with patch.object(
        client._client.embeddings,
        "create",
        side_effect=[error, _embedding_response([[0.1]])],
    ) as mock_create, patch("data.ingestion.embedder.time.sleep"):
        vectors = client.embed_batch(["text"])

    assert vectors == [[0.1]]
    assert mock_create.call_count == 2


def test_embed_batch_raises_after_max_retries() -> None:
    client = EmbeddingClient(api_key="test-key")
    error = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
    with patch.object(
        client._client.embeddings, "create", side_effect=error
    ) as mock_create, patch("data.ingestion.embedder.time.sleep"):
        with pytest.raises(APIConnectionError):
            client.embed_batch(["text"])

    assert mock_create.call_count == 3


def test_embed_chunks_batches_and_preserves_order() -> None:
    chunks = [_chunk(f"c{i}", 1, f"text {i}") for i in range(5)]

    fake_client = Mock()
    fake_client.embed_batch.side_effect = [
        [[float(i)] for i in range(2)],
        [[float(i)] for i in range(2, 4)],
        [[4.0]],
    ]

    result = embed_chunks(chunks, client=fake_client, batch_size=2)

    assert [ec.chunk.chunk_id for ec in result] == ["c0", "c1", "c2", "c3", "c4"]
    assert [ec.embedding for ec in result] == [[0.0], [1.0], [2.0], [3.0], [4.0]]
    assert fake_client.embed_batch.call_count == 3

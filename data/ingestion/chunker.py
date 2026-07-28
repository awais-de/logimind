"""Split loaded documents into overlapping, metadata-tagged chunks."""

import logging
from datetime import date

from pydantic import BaseModel

from data.ingestion.loader import LoadedDocument, load_all
from data.sources import SOURCES, DocumentSource

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


class Chunk(BaseModel):
    """A single chunk of a source document, ready for embedding.

    Attributes:
        chunk_id: Stable identifier: "<doc_id>-p<page_number>-<index>".
        doc_id: Identifier of the source document.
        doc_name: Human-readable source document title.
        category: Source document category (e.g. "rate_guide", "customs").
        region: Region the source document is specific to, if any.
        publication_date: Source document publication date, if known.
        page_number: 1-indexed page this chunk was taken from.
        text: The chunk's text content.
    """

    chunk_id: str
    doc_id: str
    doc_name: str
    category: str
    region: str | None
    publication_date: date | None
    page_number: int
    text: str


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping fixed-size windows."""
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - chunk_overlap
    windows = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        windows.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return windows


def chunk_document(
    document: LoadedDocument,
    source: DocumentSource,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split one loaded document into metadata-tagged chunks.

    Chunks stay within a single page, so each chunk keeps an accurate
    page_number rather than spanning a page boundary.

    Args:
        document: The loaded, cleaned document to split.
        source: The document's source metadata from data/sources.py.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks, in characters.

    Returns:
        Chunks in page and position order.
    """
    chunks: list[Chunk] = []
    for page in document.pages:
        for index, window in enumerate(_split_text(page.text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-p{page.page_number}-{index}",
                    doc_id=document.doc_id,
                    doc_name=source.name,
                    category=source.category,
                    region=source.region,
                    publication_date=source.publication_date,
                    page_number=page.page_number,
                    text=window,
                )
            )
    return chunks


def chunk_all(
    documents: list[LoadedDocument] | None = None,
    sources: list[DocumentSource] = SOURCES,
) -> list[Chunk]:
    """Chunk every loaded document, matching each to its source metadata.

    Args:
        documents: Loaded documents to chunk. Defaults to loading every PDF
            in data/raw/.
        sources: Source metadata to match documents against by doc_id.

    Returns:
        Chunks for every document that has a matching source entry.
    """
    if documents is None:
        documents = load_all()

    sources_by_id = {source.doc_id: source for source in sources}

    chunks: list[Chunk] = []
    for document in documents:
        source = sources_by_id.get(document.doc_id)
        if source is None:
            continue
        chunks.extend(chunk_document(document, source))
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    all_chunks = chunk_all()
    logger.info(
        "%d chunks from %d documents", len(all_chunks), len({c.doc_id for c in all_chunks})
    )

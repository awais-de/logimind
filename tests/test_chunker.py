"""Unit tests for data/ingestion/chunker.py."""

from datetime import date

from data.ingestion.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_all, chunk_document
from data.ingestion.loader import LoadedDocument, PageText
from data.sources import DocumentSource

SOURCE = DocumentSource(
    doc_id="test-doc",
    name="Test Document",
    url="https://example.com/test-doc.pdf",
    category="test",
    region="US",
    publication_date=date(2024, 1, 1),
)


def test_chunk_document_splits_long_page_with_overlap() -> None:
    long_text = "x" * (CHUNK_SIZE * 2)
    document = LoadedDocument(doc_id="test-doc", pages=[PageText(page_number=1, text=long_text)])

    chunks = chunk_document(document, SOURCE)

    assert len(chunks) > 1
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks)
    # consecutive windows overlap by CHUNK_OVERLAP characters
    assert chunks[0].text[-CHUNK_OVERLAP:] == chunks[1].text[:CHUNK_OVERLAP]


def test_chunk_document_keeps_short_page_as_single_chunk() -> None:
    document = LoadedDocument(
        doc_id="test-doc", pages=[PageText(page_number=1, text="short page text")]
    )

    chunks = chunk_document(document, SOURCE)

    assert len(chunks) == 1
    assert chunks[0].text == "short page text"
    assert chunks[0].chunk_id == "test-doc-p1-0"


def test_chunk_document_carries_source_metadata() -> None:
    document = LoadedDocument(doc_id="test-doc", pages=[PageText(page_number=3, text="content")])

    chunks = chunk_document(document, SOURCE)

    chunk = chunks[0]
    assert chunk.doc_id == "test-doc"
    assert chunk.doc_name == "Test Document"
    assert chunk.category == "test"
    assert chunk.region == "US"
    assert chunk.publication_date == date(2024, 1, 1)
    assert chunk.page_number == 3


def test_chunk_document_does_not_span_page_boundaries() -> None:
    document = LoadedDocument(
        doc_id="test-doc",
        pages=[
            PageText(page_number=1, text="page one content"),
            PageText(page_number=2, text="page two content"),
        ],
    )

    chunks = chunk_document(document, SOURCE)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2


def test_chunk_all_skips_documents_without_matching_source() -> None:
    documents = [
        LoadedDocument(doc_id="test-doc", pages=[PageText(page_number=1, text="content")]),
        LoadedDocument(doc_id="unknown-doc", pages=[PageText(page_number=1, text="content")]),
    ]

    chunks = chunk_all(documents=documents, sources=[SOURCE])

    assert {chunk.doc_id for chunk in chunks} == {"test-doc"}

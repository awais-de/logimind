"""Unit tests for data/ingestion/loader.py."""

from pathlib import Path

import fitz

from data.ingestion.loader import load_all, load_document


def _make_pdf(path: Path, page_lines: list[list[str]]) -> None:
    doc = fitz.open()
    for lines in page_lines:
        page = doc.new_page()
        for i, line in enumerate(lines):
            page.insert_text((72, 72 + i * 20), line)
    doc.save(path)
    doc.close()


def test_load_document_strips_repeated_header(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test-doc.pdf"
    _make_pdf(
        pdf_path,
        [
            ["UNCLASSIFIED (PUBLIC) HEADER", f"Unique content page {i}"]
            for i in range(1, 6)
        ],
    )

    result = load_document("test-doc", pdf_path)

    assert result.doc_id == "test-doc"
    assert len(result.pages) == 5
    for i, page in enumerate(result.pages, start=1):
        assert page.page_number == i
        assert "UNCLASSIFIED" not in page.text
        assert f"Unique content page {i}" in page.text


def test_load_document_drops_pages_with_no_usable_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test-doc.pdf"
    _make_pdf(
        pdf_path,
        [
            ["Real content on page one with enough characters to pass"],
            ["Real content on page two with enough characters to pass"],
            [],
            ["Real content on page four with enough characters to pass"],
        ],
    )

    result = load_document("test-doc", pdf_path)

    page_numbers = [page.page_number for page in result.pages]
    assert page_numbers == [1, 2, 4]


def test_load_all_reads_every_pdf_in_directory(tmp_path: Path) -> None:
    _make_pdf(tmp_path / "doc-a.pdf", [["Content for document A, plenty of chars"]])
    _make_pdf(tmp_path / "doc-b.pdf", [["Content for document B, plenty of chars"]])

    results = load_all(tmp_path)

    assert [doc.doc_id for doc in results] == ["doc-a", "doc-b"]
    assert "Content for document A" in results[0].pages[0].text

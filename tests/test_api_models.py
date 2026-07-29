"""Unit tests for api/models.py."""

import pytest
from pydantic import ValidationError

from agents.orchestrator import OrchestratorResult
from agents.planner import Plan
from agents.retriever import RetrievalResult
from api.models import QueryRequest, QueryResponse
from retrieval.semantic import SearchResult

SOURCE_KWARGS = {
    "doc_id": "test-doc",
    "doc_name": "Test Document",
    "category": "test",
    "region": None,
    "page_number": 5,
}


def _result(chunk_id: str, text: str, score: float) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=text, score=score, **SOURCE_KWARGS)


def test_query_request_requires_question() -> None:
    request = QueryRequest(question="what are the incoterms?")

    assert request.question == "what are the incoterms?"

    with pytest.raises(ValidationError):
        QueryRequest()


def test_query_response_from_orchestrator_result_with_search_results() -> None:
    plan = Plan(needs_knowledge_search=True, search_query="incoterms", needs_tracking_lookup=False)
    retrieval = RetrievalResult(search_results=[_result("c1", "x" * 300, 0.87)])
    orchestrator_result = OrchestratorResult(
        question="what are the incoterms?", plan=plan, retrieval=retrieval, answer="Here is the answer."
    )

    response = QueryResponse.from_orchestrator_result(orchestrator_result)

    assert response.answer == "Here is the answer."
    assert response.needs_knowledge_search is True
    assert response.needs_tracking_lookup is False
    assert response.tracking_info is None
    assert len(response.citations) == 1
    citation = response.citations[0]
    assert citation.doc_name == "Test Document"
    assert citation.page_number == 5
    assert citation.score == 0.87
    assert len(citation.text_snippet) == 200


def test_query_response_from_orchestrator_result_with_tracking_info() -> None:
    plan = Plan(needs_knowledge_search=False, needs_tracking_lookup=True, tracking_number="123")
    retrieval = RetrievalResult(tracking_info={"status": "Delivered"})
    orchestrator_result = OrchestratorResult(
        question="where is my package?", plan=plan, retrieval=retrieval, answer="It was delivered."
    )

    response = QueryResponse.from_orchestrator_result(orchestrator_result)

    assert response.citations == []
    assert response.tracking_info == {"status": "Delivered"}
    assert response.needs_tracking_lookup is True

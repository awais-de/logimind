"""Unit tests for retrieval/evaluation/retrieval_eval.py."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from retrieval.evaluation.retrieval_eval import (
    DEFAULT_TEST_SET_PATH,
    GroundTruthCase,
    RetrievalEvalResult,
    evaluate_case,
    load_test_set,
    record_retrieval_eval_result,
    run_retrieval_eval,
    score_context_precision,
    score_context_recall,
)
from retrieval.semantic import SearchResult

SOURCE_KWARGS = {
    "doc_id": "test-doc",
    "doc_name": "Test Document",
    "category": "test",
    "region": None,
    "page_number": 1,
}


def _result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=text, score=0.9, **SOURCE_KWARGS)


def _judge_response(payload: dict) -> Mock:
    message = Mock()
    message.content = json.dumps(payload)
    choice = Mock()
    choice.message = message
    response = Mock()
    response.choices = [choice]
    return response


def test_default_test_set_loads_and_parses() -> None:
    cases = load_test_set(DEFAULT_TEST_SET_PATH)

    assert len(cases) >= 5
    assert all(isinstance(case, GroundTruthCase) for case in cases)
    assert len({case.id for case in cases}) == len(cases)


def test_load_test_set_from_custom_path(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "question": "q",
                    "ground_truth": "a",
                    "source_doc_id": "doc",
                    "source_page": 1,
                }
            ]
        )
    )

    cases = load_test_set(path)

    assert len(cases) == 1
    assert cases[0].id == "c1"


def test_score_context_precision_returns_none_without_retrieved_chunks() -> None:
    result = score_context_precision("q", "a", [], client=Mock())

    assert result is None


def test_score_context_precision_all_relevant_is_one() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        _judge_response({"relevant": True}),
        _judge_response({"relevant": True}),
    ]
    retrieved = [_result("c1", "relevant text 1"), _result("c2", "relevant text 2")]

    result = score_context_precision("q", "a", retrieved, client)

    assert result == 1.0


def test_score_context_precision_penalizes_irrelevant_chunk_ranked_first() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        _judge_response({"relevant": False}),
        _judge_response({"relevant": True}),
    ]
    retrieved = [_result("c1", "irrelevant"), _result("c2", "relevant")]

    result = score_context_precision("q", "a", retrieved, client)

    # 1 relevant chunk, at rank 2: precision@2 = 1/2, averaged over 1 relevant chunk = 0.5
    assert result == 0.5


def test_score_context_precision_returns_zero_when_nothing_relevant() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _judge_response({"relevant": False})
    retrieved = [_result("c1", "irrelevant")]

    result = score_context_precision("q", "a", retrieved, client)

    assert result == 0.0


def test_score_context_recall_returns_none_without_retrieved_chunks() -> None:
    result = score_context_recall("a", [], client=Mock())

    assert result is None


def test_score_context_recall_computes_attributable_fraction() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _judge_response(
        {
            "claims": [
                {"claim": "DHL ships worldwide", "attributable": True},
                {"claim": "DHL was founded in 1990", "attributable": False},
            ]
        }
    )
    retrieved = [_result("c1", "DHL operates in 220 countries.")]

    result = score_context_recall("DHL ships worldwide and was founded in 1990.", retrieved, client)

    assert result == 0.5


def test_score_context_recall_returns_none_for_no_claims() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _judge_response({"claims": []})
    retrieved = [_result("c1", "some context")]

    result = score_context_recall("I don't know.", retrieved, client)

    assert result is None


def test_evaluate_case_combines_precision_and_recall() -> None:
    case = GroundTruthCase(
        id="c1", question="what are the incoterms?", ground_truth="Incoterms define responsibilities.",
        source_doc_id="doc", source_page=1,
    )
    hybrid_client = Mock()
    hybrid_client.search.return_value = [_result("c1", "Incoterms define buyer/seller responsibilities.")]

    client = Mock()
    client.chat.completions.create.side_effect = [
        _judge_response({"relevant": True}),
        _judge_response({"claims": [{"claim": "Incoterms define responsibilities", "attributable": True}]}),
    ]

    result = evaluate_case(case, hybrid_client, client)

    hybrid_client.search.assert_called_once_with("what are the incoterms?")
    assert result.case_id == "c1"
    assert result.context_precision == 1.0
    assert result.context_recall == 1.0


def test_record_and_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "eval_results.db"
    result = RetrievalEvalResult(
        case_id="c1", context_precision=0.75, context_recall=0.5,
        timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )

    record_retrieval_eval_result(result, db_path=db_path)

    connection = sqlite3.connect(db_path)
    row = connection.execute(
        "SELECT case_id, context_precision, context_recall FROM retrieval_eval_results"
    ).fetchone()
    connection.close()

    assert row == ("c1", 0.75, 0.5)


def test_record_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "eval_results.db"
    result = RetrievalEvalResult(
        case_id="c1", context_precision=1.0, context_recall=1.0,
        timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )

    record_retrieval_eval_result(result, db_path=db_path)
    record_retrieval_eval_result(result, db_path=db_path)

    connection = sqlite3.connect(db_path)
    count = connection.execute("SELECT COUNT(*) FROM retrieval_eval_results").fetchone()[0]
    connection.close()
    assert count == 1


def test_run_retrieval_eval_scores_every_case_and_records_results(tmp_path: Path) -> None:
    test_set_path = tmp_path / "cases.json"
    test_set_path.write_text(
        json.dumps(
            [
                {
                    "id": "c1", "question": "q1", "ground_truth": "a1",
                    "source_doc_id": "doc", "source_page": 1,
                },
                {
                    "id": "c2", "question": "q2", "ground_truth": "a2",
                    "source_doc_id": "doc", "source_page": 2,
                },
            ]
        )
    )
    db_path = tmp_path / "eval_results.db"

    hybrid_client = Mock()
    hybrid_client.search.return_value = [_result("c1", "some context")]

    client = Mock()
    client.chat.completions.create.side_effect = [
        _judge_response({"relevant": True}),
        _judge_response({"claims": [{"claim": "x", "attributable": True}]}),
        _judge_response({"relevant": True}),
        _judge_response({"claims": [{"claim": "x", "attributable": True}]}),
    ]

    results = run_retrieval_eval(
        test_set_path=test_set_path, hybrid_client=hybrid_client, client=client, db_path=db_path
    )

    assert [r.case_id for r in results] == ["c1", "c2"]

    connection = sqlite3.connect(db_path)
    count = connection.execute("SELECT COUNT(*) FROM retrieval_eval_results").fetchone()[0]
    connection.close()
    assert count == 2

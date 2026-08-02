"""Unit tests for monitoring/eval_loop.py."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from monitoring.eval_loop import (
    EvalResult,
    QuerySample,
    _cosine_similarity,
    evaluate_sample,
    log_query_sample,
    record_eval_result,
    run_eval_loop,
    score_answer_relevancy,
    score_faithfulness,
)


def _judge_response(payload: dict) -> Mock:
    message = Mock()
    message.content = json.dumps(payload)
    choice = Mock()
    choice.message = message
    response = Mock()
    response.choices = [choice]
    return response


def _embedding_response(vectors: list[list[float]]) -> Mock:
    response = Mock()
    response.data = [Mock(embedding=vec) for vec in vectors]
    return response


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_score_faithfulness_returns_none_without_context() -> None:
    client = Mock()
    result = score_faithfulness("q", "a", context=[], client=client)

    assert result is None
    client.chat.completions.create.assert_not_called()


def test_score_faithfulness_computes_supported_fraction() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _judge_response(
        {
            "claims": [
                {"claim": "DHL ships worldwide", "supported": True},
                {"claim": "DHL was founded in 1990", "supported": False},
            ]
        }
    )

    result = score_faithfulness(
        "does DHL ship worldwide?", "DHL ships worldwide.", ["DHL operates in 220 countries."], client
    )

    assert result == 0.5


def test_score_faithfulness_returns_none_for_no_claims() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _judge_response({"claims": []})

    result = score_faithfulness("q", "I don't know.", ["some context"], client)

    assert result is None


def test_score_answer_relevancy_averages_similarity() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _judge_response(
        {"questions": ["what is X?", "how does X work?"]}
    )
    client.embeddings.create.return_value = _embedding_response(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )

    result = score_answer_relevancy("what is X?", "X is a thing.", client)

    assert result == 0.5


def test_score_answer_relevancy_returns_none_for_empty_answer() -> None:
    client = Mock()
    result = score_answer_relevancy("q", "", client)

    assert result is None
    client.chat.completions.create.assert_not_called()


def test_evaluate_sample_combines_both_scores() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        _judge_response({"claims": [{"claim": "c", "supported": True}]}),
        _judge_response({"questions": ["q1"]}),
    ]
    client.embeddings.create.return_value = _embedding_response([[1.0, 0.0], [1.0, 0.0]])

    sample = QuerySample(
        query_id="q1",
        question="what are the incoterms?",
        answer="Incoterms define shipping responsibilities.",
        context=["Incoterms 2020 defines responsibilities."],
        timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )

    result = evaluate_sample(sample, client=client)

    assert result.query_id == "q1"
    assert result.faithfulness == 1.0
    assert result.answer_relevancy == 1.0


def test_log_and_record_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    sample = QuerySample(
        query_id="q1",
        question="what are the incoterms?",
        answer="answer text",
        context=["context a", "context b"],
        timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )

    log_query_sample(sample, db_path=db_path)

    connection = sqlite3.connect(db_path)
    row = connection.execute(
        "SELECT query_id, question, answer, context FROM query_log"
    ).fetchone()
    connection.close()

    assert row[0] == "q1"
    assert row[1] == "what are the incoterms?"
    assert json.loads(row[3]) == ["context a", "context b"]


def test_run_eval_loop_scores_only_unevaluated_samples(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    log_query_sample(
        QuerySample(
            query_id="already-scored",
            question="q",
            answer="a",
            context=["c"],
            timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
        ),
        db_path=db_path,
    )
    record_eval_result(
        EvalResult(
            query_id="already-scored",
            faithfulness=1.0,
            answer_relevancy=1.0,
            timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
        ),
        db_path=db_path,
    )
    log_query_sample(
        QuerySample(
            query_id="needs-scoring",
            question="what are the incoterms?",
            answer="answer text",
            context=["context"],
            timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
        ),
        db_path=db_path,
    )

    client = Mock()
    client.chat.completions.create.side_effect = [
        _judge_response({"claims": [{"claim": "c", "supported": True}]}),
        _judge_response({"questions": ["q1"]}),
    ]
    client.embeddings.create.return_value = _embedding_response([[1.0, 0.0], [1.0, 0.0]])

    results = run_eval_loop(sample_size=10, client=client, db_path=db_path)

    assert len(results) == 1
    assert results[0].query_id == "needs-scoring"

    connection = sqlite3.connect(db_path)
    count = connection.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0]
    connection.close()
    assert count == 2

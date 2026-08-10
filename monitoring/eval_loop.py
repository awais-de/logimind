"""Periodic RAG evaluation: faithfulness and answer-relevancy scoring.

Reimplements RAGAS's faithfulness and answer_relevancy metric definitions
via direct OpenAI calls, rather than depending on the ragas library. ragas
pulls in an old LangChain chain (langchain==0.2.17, langchain-community
<0.3.0) to import at all, which conflicts with the numpy>=2.0 required by
sentence-transformers/scipy elsewhere in this project and isn't reliably
reproducible from a fresh dependency resolution.

Every live query is logged cheaply via log_query_sample (no LLM cost).
run_eval_loop then samples unevaluated rows and scores them on demand,
since scoring costs real API spend and shouldn't run on every request.
"""

import json
import logging
import math
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import APIError, OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

EVAL_DB_PATH = Path(__file__).resolve().parents[1] / "monitoring" / "metrics.db"
JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0

FAITHFULNESS_PROMPT = """Given a question, an answer, and the context excerpts the answer was supposed to be based on, break the answer down into its individual factual claims. For each claim, decide whether it is directly supported by the context.

Question: {question}

Context excerpts:
{context}

Answer:
{answer}

Respond with ONLY a JSON object, no markdown fences and no other text:
{{"claims": [{{"claim": "...", "supported": true or false}}, ...]}}
If the answer makes no factual claims (e.g. it says it doesn't know), respond with {{"claims": []}}."""

RELEVANCY_PROMPT = """Given the following answer, generate 3 questions that this answer would be a good, direct response to. Generate diverse questions that each capture a different aspect of what the answer addresses.

Answer:
{answer}

Respond with ONLY a JSON object, no markdown fences and no other text:
{{"questions": ["...", "...", "..."]}}"""


class QuerySample(BaseModel):
    """A logged live query, ready for later evaluation.

    Attributes:
        query_id: Matches the query_id used in monitoring/metrics.py's
            step_metrics table, so a sample can be cross-referenced with
            its latency/cost data.
        question: The user's original question.
        answer: The final answer ResponseAgent produced.
        context: Text of the retrieved chunks the answer was grounded in.
            Empty if the plan didn't call for a knowledge search.
        timestamp: When the query was logged.
    """

    query_id: str
    question: str
    answer: str
    context: list[str] = []
    timestamp: datetime


class EvalResult(BaseModel):
    """Faithfulness and answer-relevancy scores for one QuerySample.

    Attributes:
        query_id: The QuerySample this result scores.
        faithfulness: Fraction of the answer's claims supported by context,
            in [0, 1]. None if the answer made no checkable claims or no
            context was retrieved.
        answer_relevancy: Average cosine similarity between the original
            question and questions generated from the answer, in [-1, 1].
            None if the answer was empty.
        timestamp: When the evaluation ran.
    """

    query_id: str
    faithfulness: float | None
    answer_relevancy: float | None
    timestamp: datetime


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS query_log (
                query_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                context TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_results (
                query_id TEXT PRIMARY KEY,
                faithfulness REAL,
                answer_relevancy REAL,
                timestamp TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def log_query_sample(sample: QuerySample, db_path: Path = EVAL_DB_PATH) -> None:
    """Persist a live query for later evaluation. Costs no API calls.

    Args:
        sample: The query to log.
        db_path: SQLite database to write to.
    """
    _init_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO query_log (query_id, question, answer, context, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                sample.query_id,
                sample.question,
                sample.answer,
                json.dumps(sample.context),
                sample.timestamp.isoformat(),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _unevaluated_samples(db_path: Path, limit: int) -> list[QuerySample]:
    _init_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT ql.query_id, ql.question, ql.answer, ql.context, ql.timestamp
            FROM query_log ql
            LEFT JOIN eval_results er ON ql.query_id = er.query_id
            WHERE er.query_id IS NULL
            ORDER BY ql.timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        connection.close()

    return [
        QuerySample(
            query_id=row[0],
            question=row[1],
            answer=row[2],
            context=json.loads(row[3]),
            timestamp=datetime.fromisoformat(row[4]),
        )
        for row in rows
    ]


def record_eval_result(result: EvalResult, db_path: Path = EVAL_DB_PATH) -> None:
    """Persist an evaluation result.

    Args:
        result: The scored result to store.
        db_path: SQLite database to write to.
    """
    _init_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO eval_results (query_id, faithfulness, answer_relevancy, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (
                result.query_id,
                result.faithfulness,
                result.answer_relevancy,
                result.timestamp.isoformat(),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    logger.info(
        "Recorded eval: query=%s faithfulness=%s answer_relevancy=%s",
        result.query_id, result.faithfulness, result.answer_relevancy,
    )


def call_judge_json(client: OpenAI, prompt: str) -> dict:
    """Call the judge model with a prompt expecting a JSON object reply.

    Shared by any module scoring something via an LLM judge (this module's
    faithfulness/relevancy scoring, and monitoring/guardrail_eval.py's
    adversarial probe scoring) so the retry/backoff and JSON-extraction
    logic isn't duplicated.

    Args:
        client: OpenAI client to call.
        prompt: Prompt instructing the model to reply with a JSON object.

    Returns:
        The parsed JSON object from the model's response.

    Raises:
        ValueError: The response contained no JSON object.
        openai.APIError: The call failed after all retries.
    """
    last_error: APIError | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in judge response: {content!r}")
            return json.loads(match.group(0))
        except APIError as exc:
            last_error = exc
            wait = BACKOFF_SECONDS * (2**attempt)
            logger.warning(
                "Judge request failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt + 1, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)

    raise last_error


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_faithfulness(
    question: str, answer: str, context: list[str], client: OpenAI
) -> float | None:
    """Score how well an answer's claims are supported by retrieved context.

    Args:
        question: The user's original question.
        answer: The answer to check.
        context: Retrieved excerpts the answer should be grounded in.
        client: OpenAI client to use as the judge.

    Returns:
        Fraction of claims supported by context, in [0, 1]. None if the
        answer made no checkable claims, or no context was retrieved.
    """
    if not context:
        return None

    prompt = FAITHFULNESS_PROMPT.format(
        question=question, context="\n\n".join(context), answer=answer
    )
    parsed = call_judge_json(client, prompt)
    claims = parsed.get("claims", [])
    if not claims:
        return None

    supported = sum(1 for claim in claims if claim.get("supported"))
    return supported / len(claims)


def score_answer_relevancy(question: str, answer: str, client: OpenAI) -> float | None:
    """Score how relevant an answer is to the question that was asked.

    Generates candidate questions the answer would suit, then averages the
    embedding cosine similarity between each candidate and the real
    question. Higher means the answer stayed on-topic.

    Args:
        question: The user's original question.
        answer: The answer to check.
        client: OpenAI client used for both the judge call and embeddings.

    Returns:
        Average cosine similarity in [-1, 1]. None if the answer is empty.
    """
    if not answer.strip():
        return None

    prompt = RELEVANCY_PROMPT.format(answer=answer)
    parsed = call_judge_json(client, prompt)
    generated_questions = parsed.get("questions", [])
    if not generated_questions:
        return None

    embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL, input=[question, *generated_questions]
    )
    vectors = [item.embedding for item in embedding_response.data]
    question_vector, generated_vectors = vectors[0], vectors[1:]

    similarities = [_cosine_similarity(question_vector, vec) for vec in generated_vectors]
    return sum(similarities) / len(similarities)


def evaluate_sample(sample: QuerySample, client: OpenAI | None = None) -> EvalResult:
    """Score a single QuerySample for faithfulness and answer relevancy.

    Args:
        sample: The query/answer/context to evaluate.
        client: OpenAI client to use. Defaults to a new OpenAI().

    Returns:
        The scored EvalResult.
    """
    if client is None:
        client = OpenAI()

    faithfulness = score_faithfulness(sample.question, sample.answer, sample.context, client)
    answer_relevancy = score_answer_relevancy(sample.question, sample.answer, client)

    return EvalResult(
        query_id=sample.query_id,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        timestamp=datetime.now(timezone.utc),
    )


def run_eval_loop(
    sample_size: int = 10,
    client: OpenAI | None = None,
    db_path: Path = EVAL_DB_PATH,
) -> list[EvalResult]:
    """Score a batch of unevaluated logged queries and record the results.

    Meant to be invoked periodically (e.g. on a schedule or manually), not
    on every request, since each sample costs real judge-model and
    embedding-model API calls.

    Args:
        sample_size: Maximum number of unevaluated queries to score.
        client: OpenAI client to use. Defaults to a new OpenAI().
        db_path: SQLite database to read samples from and write results to.

    Returns:
        The EvalResults produced this run, in the same order as the
        samples were pulled (most recently logged first).
    """
    if client is None:
        client = OpenAI()

    samples = _unevaluated_samples(db_path, limit=sample_size)
    results = []
    for sample in samples:
        result = evaluate_sample(sample, client=client)
        record_eval_result(result, db_path=db_path)
        results.append(result)

    logger.info("Eval loop scored %d samples", len(results))
    return results

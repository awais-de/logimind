"""Retrieval-only evaluation: context precision and context recall.

Reimplements RAGAS's context_precision and context_recall metric
definitions via direct OpenAI calls, rather than depending on the ragas
library (see monitoring/eval_loop.py's docstring for why: ragas forces a
broken numpy/langchain dependency chain).

This module deliberately does not import from monitoring/ (a higher
layer than retrieval/ -- see CLAUDE.md's "no layer imports from a layer
above it" rule) even though monitoring/eval_loop.py's call_judge_json
does the same retry/backoff/JSON-extraction work. The judge-calling
helper below is a small, intentional duplicate, kept local to this layer.

Runs separately from pytest, on demand, since scoring costs real judge
and embedding API spend: `python -m retrieval.evaluation.retrieval_eval`.

test_set.json ships with a small seed of hand-verified Q/A/source-page
triples (pulled directly from the ingested PDFs) covering three
documents. Growing it toward a real ground-truth set (80-200 cases,
broader document coverage) is still open -- add cases directly to
test_set.json in the same shape.
"""

import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import APIError, OpenAI
from pydantic import BaseModel

from retrieval.hybrid import HybridSearchClient
from retrieval.semantic import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_TEST_SET_PATH = Path(__file__).resolve().parent / "test_set.json"
RETRIEVAL_EVAL_DB_PATH = Path(__file__).resolve().parent / "eval_results.db"
JUDGE_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0

RELEVANCE_PROMPT = """Given a question, a reference answer, and one retrieved context excerpt, decide whether the excerpt is relevant to answering the question -- i.e. whether it contains information that helps produce the reference answer.

Question: {question}

Reference answer: {ground_truth}

Retrieved excerpt:
{chunk_text}

Respond with ONLY a JSON object, no markdown fences and no other text:
{{"relevant": true or false}}"""

CONTEXT_RECALL_PROMPT = """Given a reference answer and a set of retrieved context excerpts, break the reference answer down into its individual factual claims. For each claim, decide whether it can be attributed to (is supported by) the retrieved context.

Reference answer: {ground_truth}

Retrieved context:
{context}

Respond with ONLY a JSON object, no markdown fences and no other text:
{{"claims": [{{"claim": "...", "attributable": true or false}}, ...]}}
If the reference answer makes no factual claims, respond with {{"claims": []}}."""


class GroundTruthCase(BaseModel):
    """One curated question/answer/source case from test_set.json.

    Attributes:
        id: Stable identifier for this case.
        question: The natural-language question to retrieve for.
        ground_truth: The reference answer, used as the standard context
            precision/recall are scored against.
        source_doc_id: doc_id (per data/sources.py) the ground_truth was
            drawn from, for human traceability.
        source_page: Page number within source_doc_id the ground_truth
            was drawn from.
    """

    id: str
    question: str
    ground_truth: str
    source_doc_id: str
    source_page: int


class RetrievalEvalResult(BaseModel):
    """Context precision/recall scores for one GroundTruthCase.

    Attributes:
        case_id: The GroundTruthCase this result scores.
        context_precision: Rank-weighted precision of the retrieved
            chunks, in [0, 1]. None if nothing was retrieved.
        context_recall: Fraction of the ground truth's claims attributable
            to the retrieved context, in [0, 1]. None if nothing was
            retrieved, or the ground truth made no checkable claims.
        timestamp: When the evaluation ran.
    """

    case_id: str
    context_precision: float | None
    context_recall: float | None
    timestamp: datetime


def load_test_set(path: Path = DEFAULT_TEST_SET_PATH) -> list[GroundTruthCase]:
    """Load the curated ground-truth cases from test_set.json.

    Args:
        path: Path to the test set JSON file.

    Returns:
        The parsed ground-truth cases.
    """
    with open(path) as f:
        raw_cases = json.load(f)
    return [GroundTruthCase.model_validate(raw_case) for raw_case in raw_cases]


def _call_judge_json(client: OpenAI, prompt: str) -> dict:
    """Call the judge model expecting a JSON object reply, with retries.

    Deliberately duplicated from monitoring/eval_loop.py's call_judge_json
    rather than imported -- see this module's docstring.
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


def score_context_precision(
    question: str, ground_truth: str, retrieved: list[SearchResult], client: OpenAI
) -> float | None:
    """Score how precisely the retrieved chunks are ranked, RAGAS-style.

    Each chunk gets a binary relevance verdict from the judge, then
    precision is computed as rank-weighted average precision: for each
    relevant chunk at rank k, precision@k contributes to the average,
    normalized by the total number of relevant chunks. This rewards
    relevant chunks appearing earlier, not just being present.

    Args:
        question: The question that was searched for.
        ground_truth: The reference answer the chunks should support.
        retrieved: Chunks returned by HybridSearchClient.search, in
            ranked order.
        client: OpenAI client to use as the judge.

    Returns:
        Average precision in [0, 1]. None if nothing was retrieved.
    """
    if not retrieved:
        return None

    verdicts = []
    for chunk in retrieved:
        prompt = RELEVANCE_PROMPT.format(question=question, ground_truth=ground_truth, chunk_text=chunk.text)
        parsed = _call_judge_json(client, prompt)
        verdicts.append(bool(parsed.get("relevant")))

    total_relevant = sum(verdicts)
    if total_relevant == 0:
        return 0.0

    relevant_so_far = 0
    precision_sum = 0.0
    for rank, relevant in enumerate(verdicts, start=1):
        if relevant:
            relevant_so_far += 1
            precision_sum += relevant_so_far / rank

    return precision_sum / total_relevant


def score_context_recall(ground_truth: str, retrieved: list[SearchResult], client: OpenAI) -> float | None:
    """Score how much of the ground truth is covered by retrieved context.

    Breaks the ground truth answer into claims, then checks how many are
    attributable to the retrieved context -- unlike context precision,
    this doesn't require an exact chunk match, just that the information
    is present somewhere in what was retrieved.

    Args:
        ground_truth: The reference answer to check coverage of.
        retrieved: Chunks returned by HybridSearchClient.search.
        client: OpenAI client to use as the judge.

    Returns:
        Fraction of claims attributable to the retrieved context, in
        [0, 1]. None if nothing was retrieved, or the ground truth made
        no checkable claims.
    """
    if not retrieved:
        return None

    context = "\n\n".join(chunk.text for chunk in retrieved)
    prompt = CONTEXT_RECALL_PROMPT.format(ground_truth=ground_truth, context=context)
    parsed = _call_judge_json(client, prompt)
    claims = parsed.get("claims", [])
    if not claims:
        return None

    attributable = sum(1 for claim in claims if claim.get("attributable"))
    return attributable / len(claims)


def evaluate_case(
    case: GroundTruthCase, hybrid_client: HybridSearchClient, client: OpenAI
) -> RetrievalEvalResult:
    """Retrieve for one GroundTruthCase and score precision/recall.

    Args:
        case: The ground-truth case to evaluate.
        hybrid_client: Client to retrieve chunks with.
        client: OpenAI client to use as the judge.

    Returns:
        The scored RetrievalEvalResult.
    """
    retrieved = hybrid_client.search(case.question)
    precision = score_context_precision(case.question, case.ground_truth, retrieved, client)
    recall = score_context_recall(case.ground_truth, retrieved, client)

    return RetrievalEvalResult(
        case_id=case.id,
        context_precision=precision,
        context_recall=recall,
        timestamp=datetime.now(timezone.utc),
    )


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_eval_results (
                case_id TEXT PRIMARY KEY,
                context_precision REAL,
                context_recall REAL,
                timestamp TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def record_retrieval_eval_result(result: RetrievalEvalResult, db_path: Path = RETRIEVAL_EVAL_DB_PATH) -> None:
    """Persist one case's precision/recall scores.

    Args:
        result: The scored result to store.
        db_path: SQLite database to write to.
    """
    _init_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO retrieval_eval_results (case_id, context_precision, context_recall, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (result.case_id, result.context_precision, result.context_recall, result.timestamp.isoformat()),
        )
        connection.commit()
    finally:
        connection.close()
    logger.info(
        "Recorded retrieval eval: case=%s precision=%s recall=%s",
        result.case_id, result.context_precision, result.context_recall,
    )


def run_retrieval_eval(
    test_set_path: Path = DEFAULT_TEST_SET_PATH,
    hybrid_client: HybridSearchClient | None = None,
    client: OpenAI | None = None,
    db_path: Path = RETRIEVAL_EVAL_DB_PATH,
) -> list[RetrievalEvalResult]:
    """Run every case in the ground-truth test set and record the results.

    Meant to be invoked periodically/manually, not on every request: each
    case costs real retrieval (embedding + reranker) and judge-model API
    calls -- one judge call per retrieved chunk for precision, plus one
    for recall.

    Args:
        test_set_path: Path to the ground-truth test set JSON file.
        hybrid_client: Client to retrieve chunks with. Defaults to a new
            HybridSearchClient.
        client: OpenAI client to use as the judge. Defaults to a new
            OpenAI().
        db_path: SQLite database to record results in.

    Returns:
        One RetrievalEvalResult per test-set case, in file order.
    """
    if hybrid_client is None:
        hybrid_client = HybridSearchClient()
    if client is None:
        client = OpenAI()

    cases = load_test_set(test_set_path)
    results = []
    for case in cases:
        result = evaluate_case(case, hybrid_client, client)
        record_retrieval_eval_result(result, db_path=db_path)
        results.append(result)

    logger.info("Retrieval eval scored %d cases", len(results))
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    run_retrieval_eval()

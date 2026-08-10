"""Orchestrator: wires PlannerAgent, RetrieverAgent, ResponseAgent into one pipeline."""

import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from autogen_core.models import ChatCompletionClient, RequestUsage
from langsmith import traceable
from pydantic import BaseModel

from agents.model_client import DEFAULT_MODEL, build_claude_client
from agents.planner import Plan, build_planner_agent, run_planner
from agents.responder import build_responder_agent, run_responder
from agents.retriever import RetrievalResult, run_retriever
from monitoring.eval_loop import EVAL_DB_PATH, QuerySample, log_query_sample
from monitoring.metrics import METRICS_DB_PATH, StepMetric, compute_cost, record_step
from retrieval.cache import SemanticCache

logger = logging.getLogger(__name__)


class OrchestratorResult(BaseModel):
    """Full trace of one query through the agent pipeline.

    Attributes:
        question: The original user question.
        plan: PlannerAgent's decision.
        retrieval: What RetrieverAgent found.
        answer: ResponseAgent's final synthesized answer.
    """

    question: str
    plan: Plan
    retrieval: RetrievalResult
    answer: str


class Orchestrator:
    """Runs a user question through Planner -> Retriever -> Responder.

    This is a fixed sequential pipeline rather than a free-form AutoGen
    group chat: PlannerAgent's decision fully determines what
    RetrieverAgent does, so there's no open-ended conversation between
    agents to manage, just a deterministic three-step handoff. LangSmith
    tracing is enabled via @traceable; it silently no-ops until
    LANGSMITH_API_KEY / LANGSMITH_TRACING are configured.
    """

    def __init__(
        self,
        model_client: ChatCompletionClient | None = None,
        model: str = DEFAULT_MODEL,
        metrics_db_path: Path = METRICS_DB_PATH,
        eval_db_path: Path = EVAL_DB_PATH,
        cache: SemanticCache | None = None,
    ) -> None:
        """Build the pipeline's agents.

        Args:
            model_client: Claude client shared by PlannerAgent and
                ResponseAgent. Defaults to a new client via
                build_claude_client().
            model: Model ID used for cost attribution in recorded metrics.
                Only meaningful when model_client is left as default.
            metrics_db_path: SQLite database step metrics are written to.
            eval_db_path: SQLite database query samples are logged to, for
                later RAGAS-style evaluation via monitoring/eval_loop.py.
            cache: Semantic cache checked before running the pipeline, and
                populated after. Defaults to a new SemanticCache().
        """
        self._model_client = model_client or build_claude_client(model=model)
        self._model = model
        self._metrics_db_path = metrics_db_path
        self._eval_db_path = eval_db_path
        self._cache = cache or SemanticCache()
        self._planner = build_planner_agent(self._model_client)
        self._responder = build_responder_agent(self._model_client)

    @traceable(name="LogiMind.ask", run_type="chain")
    async def ask(self, question: str) -> OrchestratorResult:
        """Answer a user question end-to-end.

        Checks the semantic cache first: a repeated or near-duplicate
        question short-circuits the full Planner -> Retriever -> Responder
        pipeline and returns the cached result instead.

        Args:
            question: The user's question.

        Returns:
            The full pipeline trace: the plan, what was retrieved, and the
            final answer. Identical in shape whether served from cache or
            computed fresh.
        """
        query_id = str(uuid.uuid4())

        start = time.perf_counter()
        cached = self._cache.get(question)
        if cached is not None:
            result = OrchestratorResult.model_validate_json(cached)
            self._record_step(query_id, "cache_hit", start, usage=None)
            self._log_query_sample(query_id, question, result.answer, result.retrieval)
            return result

        start = time.perf_counter()
        plan, planner_usage = await run_planner(self._planner, question)
        self._record_step(query_id, "planner", start, planner_usage)

        start = time.perf_counter()
        retrieval = run_retriever(plan)
        self._record_step(query_id, "retriever", start, usage=None)

        start = time.perf_counter()
        answer, responder_usage = await run_responder(self._responder, question, retrieval)
        self._record_step(query_id, "responder", start, responder_usage)

        result = OrchestratorResult(question=question, plan=plan, retrieval=retrieval, answer=answer)
        self._cache.set(question, result.model_dump_json())

        self._log_query_sample(query_id, question, answer, retrieval)

        return result

    def _record_step(
        self, query_id: str, step: str, start_time: float, usage: RequestUsage | None
    ) -> None:
        """Time and record one pipeline step. Never raises -- a metrics
        write failure shouldn't break the actual query response.
        """
        latency_ms = (time.perf_counter() - start_time) * 1000
        cost_usd = None
        if usage is not None:
            cost_usd = compute_cost(self._model, usage.prompt_tokens, usage.completion_tokens)

        metric = StepMetric(
            query_id=query_id,
            step=step,
            latency_ms=latency_ms,
            model=self._model if usage is not None else None,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=usage.completion_tokens if usage is not None else None,
            cost_usd=cost_usd,
            timestamp=datetime.now(timezone.utc),
        )
        try:
            record_step(metric, db_path=self._metrics_db_path)
        except sqlite3.Error as exc:
            logger.error("Failed to record metrics for step %s: %s", step, exc)

    def _log_query_sample(
        self, query_id: str, question: str, answer: str, retrieval: RetrievalResult
    ) -> None:
        """Log a query for later evaluation. Never raises -- a logging
        failure shouldn't break the actual query response. Costs no API
        calls; scoring happens separately via monitoring/eval_loop.py.
        """
        sample = QuerySample(
            query_id=query_id,
            question=question,
            answer=answer,
            context=[result.text for result in retrieval.search_results],
            timestamp=datetime.now(timezone.utc),
        )
        try:
            log_query_sample(sample, db_path=self._eval_db_path)
        except sqlite3.Error as exc:
            logger.error("Failed to log query sample %s: %s", query_id, exc)

    async def close(self) -> None:
        """Release the underlying model client's resources."""
        await self._model_client.close()

"""Orchestrator: wires PlannerAgent, RetrieverAgent, ResponseAgent into one pipeline."""

import logging

from autogen_core.models import ChatCompletionClient
from langsmith import traceable
from pydantic import BaseModel

from agents.model_client import build_claude_client
from agents.planner import Plan, build_planner_agent, run_planner
from agents.responder import build_responder_agent, run_responder
from agents.retriever import RetrievalResult, run_retriever

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

    def __init__(self, model_client: ChatCompletionClient | None = None) -> None:
        """Build the pipeline's agents.

        Args:
            model_client: Claude client shared by PlannerAgent and
                ResponseAgent. Defaults to a new client via
                build_claude_client().
        """
        self._model_client = model_client or build_claude_client()
        self._planner = build_planner_agent(self._model_client)
        self._responder = build_responder_agent(self._model_client)

    @traceable(name="LogiMind.ask", run_type="chain")
    async def ask(self, question: str) -> OrchestratorResult:
        """Answer a user question end-to-end.

        Args:
            question: The user's question.

        Returns:
            The full pipeline trace: the plan, what was retrieved, and the
            final answer.
        """
        plan = await run_planner(self._planner, question)
        retrieval = run_retriever(plan)
        answer = await run_responder(self._responder, question, retrieval)
        return OrchestratorResult(question=question, plan=plan, retrieval=retrieval, answer=answer)

    async def close(self) -> None:
        """Release the underlying model client's resources."""
        await self._model_client.close()

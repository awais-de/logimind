"""RetrieverAgent: executes the PlannerAgent's plan using DHL tools."""

import logging

from pydantic import BaseModel

from agents.planner import Plan
from agents.tools.dhl_search import dhl_knowledge_search
from agents.tools.dhl_tracking_mock import dhl_tracking_mock
from retrieval.semantic import SearchResult

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """Everything RetrieverAgent gathered for a Plan.

    Attributes:
        search_results: Chunks found via dhl_knowledge_search, if the plan
            requested a knowledge search.
        tracking_info: Mock tracking status, if the plan requested a
            tracking lookup.
    """

    search_results: list[SearchResult] = []
    tracking_info: dict | None = None


def run_retriever(plan: Plan) -> RetrievalResult:
    """Execute a PlannerAgent Plan by calling the tools it specifies.

    This deliberately isn't an LLM-backed AutoGen agent: PlannerAgent has
    already decided what's needed, so executing that decision is a plain,
    deterministic dispatch rather than something requiring another model
    call.

    Args:
        plan: The plan produced by PlannerAgent.

    Returns:
        Whatever knowledge-search results and/or tracking info the plan
        called for. Both fields are empty/None if the plan needed neither.
    """
    search_results: list[SearchResult] = []
    tracking_info: dict | None = None

    if plan.needs_knowledge_search and plan.search_query:
        search_results = dhl_knowledge_search(plan.search_query)

    if plan.needs_tracking_lookup and plan.tracking_number:
        tracking_info = dhl_tracking_mock(plan.tracking_number)

    return RetrievalResult(search_results=search_results, tracking_info=tracking_info)

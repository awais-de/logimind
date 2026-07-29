"""The /query endpoint: routes a question through the agent pipeline."""

import logging

from fastapi import APIRouter, Request

from agents.orchestrator import Orchestrator
from api.models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    """Answer a user question via the full agent pipeline.

    Args:
        request: The user's question.
        http_request: The incoming request, used to access the shared
            Orchestrator stored on app.state.

    Returns:
        The synthesized answer with citations.
    """
    orchestrator: Orchestrator = http_request.app.state.orchestrator
    result = await orchestrator.ask(request.question)
    return QueryResponse.from_orchestrator_result(result)

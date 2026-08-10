"""Pydantic request/response models for the /query and /feedback endpoints."""

from pydantic import BaseModel, Field

from agents.orchestrator import OrchestratorResult
from monitoring.feedback import Vote

SNIPPET_LENGTH = 200
MAX_QUESTION_LENGTH = 500


class QueryRequest(BaseModel):
    """A user's question sent to the /query endpoint.

    Attributes:
        question: The user's natural-language question. Length-capped
            since each request spends real OpenAI/Anthropic API budget.
    """

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


class Citation(BaseModel):
    """A single source citation for the answer.

    Attributes:
        doc_name: Source document title.
        page_number: Page the cited chunk was taken from.
        score: The chunk's relevance score.
        text_snippet: A short preview of the cited chunk's text.
    """

    doc_name: str
    page_number: int
    score: float
    text_snippet: str


class QueryResponse(BaseModel):
    """The API's response to a /query request.

    Attributes:
        query_id: Identifies this specific request -- pass it back to
            POST /feedback to vote on the answer.
        answer: The synthesized answer, with inline citations.
        citations: Source documents/pages the answer draws from.
        needs_knowledge_search: Whether the query triggered a knowledge
            search.
        needs_tracking_lookup: Whether the query triggered a tracking
            lookup.
        tracking_info: Mock tracking status, if a tracking lookup was made.
    """

    query_id: str
    answer: str
    citations: list[Citation]
    needs_knowledge_search: bool
    needs_tracking_lookup: bool
    tracking_info: dict | None = None

    @classmethod
    def from_orchestrator_result(cls, result: OrchestratorResult) -> "QueryResponse":
        """Build a QueryResponse from an OrchestratorResult.

        Args:
            result: The full pipeline result to summarize for the API.

        Returns:
            A QueryResponse with citations extracted from the retrieved
            chunks.
        """
        citations = [
            Citation(
                doc_name=r.doc_name,
                page_number=r.page_number,
                score=r.score,
                text_snippet=r.text[:SNIPPET_LENGTH],
            )
            for r in result.retrieval.search_results
        ]
        return cls(
            query_id=result.query_id,
            answer=result.answer,
            citations=citations,
            needs_knowledge_search=any(s.tool == "knowledge_search" for s in result.plan.steps),
            needs_tracking_lookup=any(s.tool == "tracking_lookup" for s in result.plan.steps),
            tracking_info=result.retrieval.tracking_info,
        )


class FeedbackRequest(BaseModel):
    """A thumbs up/down vote on a /query response.

    Attributes:
        query_id: The query_id from the QueryResponse being voted on.
        vote: "up" or "down".
    """

    query_id: str
    vote: Vote

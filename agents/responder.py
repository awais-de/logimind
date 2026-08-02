"""ResponseAgent: synthesizes the final answer with citations."""

import logging

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient, RequestUsage

from agents.retriever import RetrievalResult
from monitoring.prompt_versions.responder import RESPONDER_SYSTEM_PROMPT_V1

logger = logging.getLogger(__name__)


def build_responder_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    """Build the ResponseAgent.

    Args:
        model_client: The chat completion client the agent uses.

    Returns:
        An AssistantAgent that synthesizes a final answer with citations
        from retrieved context.
    """
    return AssistantAgent(
        name="ResponseAgent",
        model_client=model_client,
        system_message=RESPONDER_SYSTEM_PROMPT_V1,
    )


def _format_context(question: str, retrieval: RetrievalResult) -> str:
    """Format the user's question and retrieved context into one prompt."""
    parts = [f"User question: {question}", ""]

    if retrieval.search_results:
        parts.append("Relevant document excerpts:")
        for result in retrieval.search_results:
            parts.append(f"[Source: {result.doc_name}, p.{result.page_number}]\n{result.text}")
    else:
        parts.append("No relevant document excerpts were found.")

    parts.append("")

    if retrieval.tracking_info:
        parts.append("Tracking status:")
        for key, value in retrieval.tracking_info.items():
            parts.append(f"{key}: {value}")
    else:
        parts.append("No tracking information was requested or found.")

    return "\n".join(parts)


async def run_responder(
    agent: AssistantAgent, question: str, retrieval: RetrievalResult
) -> tuple[str, RequestUsage | None]:
    """Run the ResponseAgent to synthesize a final answer.

    Args:
        agent: A ResponseAgent built by build_responder_agent.
        question: The original user question.
        retrieval: What RetrieverAgent found for the question.

    Returns:
        A tuple of the synthesized answer text (with citations) and the
        model's token usage for this call (None if unreported), for cost
        tracking.
    """
    context = _format_context(question, retrieval)
    result = await agent.run(task=context)
    last_message = result.messages[-1]
    return last_message.content, last_message.models_usage

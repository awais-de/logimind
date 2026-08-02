"""Unit tests for agents/planner.py."""

from unittest.mock import AsyncMock, Mock

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import RequestUsage

from agents.model_client import build_claude_client
from agents.planner import Plan, build_planner_agent, parse_plan, run_planner


def test_build_planner_agent_returns_named_assistant_agent() -> None:
    client = build_claude_client(api_key="dummy-test-key")

    agent = build_planner_agent(client)

    assert isinstance(agent, AssistantAgent)
    assert agent.name == "PlannerAgent"


def test_plan_allows_no_action_needed() -> None:
    plan = Plan(needs_knowledge_search=False, needs_tracking_lookup=False)

    assert plan.search_query is None
    assert plan.tracking_number is None


def test_plan_requires_the_two_needs_fields() -> None:
    plan = Plan(
        needs_knowledge_search=True,
        search_query="what are the incoterms",
        needs_tracking_lookup=False,
    )

    assert plan.needs_knowledge_search is True
    assert plan.search_query == "what are the incoterms"


def test_parse_plan_from_plain_json() -> None:
    raw = '{"needs_knowledge_search": true, "search_query": "incoterms", "needs_tracking_lookup": false, "tracking_number": null}'

    plan = parse_plan(raw)

    assert plan.needs_knowledge_search is True
    assert plan.search_query == "incoterms"
    assert plan.needs_tracking_lookup is False


def test_parse_plan_strips_markdown_fence() -> None:
    raw = '```json\n{"needs_knowledge_search": false, "needs_tracking_lookup": true, "tracking_number": "123", "search_query": null}\n```'

    plan = parse_plan(raw)

    assert plan.needs_tracking_lookup is True
    assert plan.tracking_number == "123"


def test_parse_plan_raises_on_no_json() -> None:
    with pytest.raises(ValueError, match="No JSON object found"):
        parse_plan("I'm not sure how to answer that.")


def test_parse_plan_raises_on_schema_mismatch() -> None:
    with pytest.raises(ValueError, match="did not match Plan schema"):
        parse_plan('{"foo": "bar"}')


@pytest.mark.asyncio
async def test_run_planner_parses_agent_response_and_returns_usage() -> None:
    fake_message = Mock()
    fake_message.content = (
        '{"needs_knowledge_search": true, "search_query": "packing rules", '
        '"needs_tracking_lookup": false, "tracking_number": null}'
    )
    fake_message.models_usage = RequestUsage(prompt_tokens=120, completion_tokens=40)
    fake_result = Mock()
    fake_result.messages = [fake_message]

    fake_agent = Mock()
    fake_agent.run = AsyncMock(return_value=fake_result)

    plan, usage = await run_planner(fake_agent, "how should I pack fragile items?")

    fake_agent.run.assert_awaited_once_with(task="how should I pack fragile items?")
    assert plan.needs_knowledge_search is True
    assert plan.search_query == "packing rules"
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 40

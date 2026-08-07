"""Unit tests for agents/planner.py."""

from unittest.mock import AsyncMock, Mock

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import RequestUsage

from agents.model_client import build_claude_client
from agents.planner import Plan, Step, build_planner_agent, parse_plan, run_planner


def test_build_planner_agent_returns_named_assistant_agent() -> None:
    client = build_claude_client(api_key="dummy-test-key")

    agent = build_planner_agent(client)

    assert isinstance(agent, AssistantAgent)
    assert agent.name == "PlannerAgent"


def test_plan_allows_no_steps() -> None:
    plan = Plan()

    assert plan.steps == []


def test_plan_holds_ordered_steps() -> None:
    plan = Plan(
        steps=[
            Step(tool="tracking_lookup", tracking_number="123"),
            Step(tool="knowledge_search", search_query="customs rules for {{step_1.destination}}"),
        ]
    )

    assert len(plan.steps) == 2
    assert plan.steps[0].tool == "tracking_lookup"
    assert plan.steps[1].search_query == "customs rules for {{step_1.destination}}"


def test_parse_plan_from_plain_json() -> None:
    raw = '{"steps": [{"tool": "knowledge_search", "search_query": "incoterms", "tracking_number": null}]}'

    plan = parse_plan(raw)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "knowledge_search"
    assert plan.steps[0].search_query == "incoterms"


def test_parse_plan_strips_markdown_fence() -> None:
    raw = '```json\n{"steps": [{"tool": "tracking_lookup", "tracking_number": "123", "search_query": null}]}\n```'

    plan = parse_plan(raw)

    assert plan.steps[0].tool == "tracking_lookup"
    assert plan.steps[0].tracking_number == "123"


def test_parse_plan_handles_no_steps_needed() -> None:
    raw = '{"steps": []}'

    plan = parse_plan(raw)

    assert plan.steps == []


def test_parse_plan_handles_ordered_multi_step_plan_with_dependency() -> None:
    raw = (
        '{"steps": ['
        '{"tool": "tracking_lookup", "tracking_number": "1234567890", "search_query": null}, '
        '{"tool": "knowledge_search", "search_query": "customs rules for {{step_1.destination}}", "tracking_number": null}'
        "]}"
    )

    plan = parse_plan(raw)

    assert len(plan.steps) == 2
    assert plan.steps[0].tool == "tracking_lookup"
    assert plan.steps[0].tracking_number == "1234567890"
    assert plan.steps[1].tool == "knowledge_search"
    assert plan.steps[1].search_query == "customs rules for {{step_1.destination}}"


def test_parse_plan_raises_on_no_json() -> None:
    with pytest.raises(ValueError, match="No JSON object found"):
        parse_plan("I'm not sure how to answer that.")


def test_parse_plan_raises_on_schema_mismatch() -> None:
    with pytest.raises(ValueError, match="did not match Plan schema"):
        parse_plan('{"steps": [{"tool": "not_a_real_tool"}]}')


@pytest.mark.asyncio
async def test_run_planner_parses_agent_response_and_returns_usage() -> None:
    fake_message = Mock()
    fake_message.content = '{"steps": [{"tool": "knowledge_search", "search_query": "packing rules", "tracking_number": null}]}'
    fake_message.models_usage = RequestUsage(prompt_tokens=120, completion_tokens=40)
    fake_result = Mock()
    fake_result.messages = [fake_message]

    fake_agent = Mock()
    fake_agent.run = AsyncMock(return_value=fake_result)

    plan, usage = await run_planner(fake_agent, "how should I pack fragile items?")

    fake_agent.run.assert_awaited_once_with(task="how should I pack fragile items?")
    assert plan.steps[0].tool == "knowledge_search"
    assert plan.steps[0].search_query == "packing rules"
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 40


@pytest.mark.asyncio
async def test_run_planner_parses_ordered_multi_step_response() -> None:
    fake_message = Mock()
    fake_message.content = (
        '{"steps": ['
        '{"tool": "tracking_lookup", "tracking_number": "1234567890", "search_query": null}, '
        '{"tool": "knowledge_search", "search_query": "customs rules for {{step_1.destination}}", "tracking_number": null}'
        "]}"
    )
    fake_message.models_usage = None
    fake_result = Mock()
    fake_result.messages = [fake_message]

    fake_agent = Mock()
    fake_agent.run = AsyncMock(return_value=fake_result)

    plan, _ = await run_planner(fake_agent, "where's my package and what customs rules apply there?")

    assert [step.tool for step in plan.steps] == ["tracking_lookup", "knowledge_search"]
    assert plan.steps[1].search_query == "customs rules for {{step_1.destination}}"

"""Unit tests for agents/planner.py."""

from autogen_agentchat.agents import AssistantAgent

from agents.model_client import build_claude_client
from agents.planner import Plan, build_planner_agent


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

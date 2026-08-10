"""Unit tests for agents/model_client.py."""

from autogen_ext.models.anthropic import AnthropicChatCompletionClient

from agents.model_client import (
    CHEAP_MODEL,
    DEFAULT_MODEL,
    STRONG_MODEL,
    build_claude_client,
    select_response_model,
)
from agents.planner import Plan, Step


def test_build_claude_client_with_explicit_args() -> None:
    client = build_claude_client(model="claude-opus-5", api_key="dummy-test-key")

    assert isinstance(client, AnthropicChatCompletionClient)


def test_build_claude_client_reads_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-test-key")

    client = build_claude_client()

    assert isinstance(client, AnthropicChatCompletionClient)


def test_build_claude_client_uses_default_model_name() -> None:
    client = build_claude_client(api_key="dummy-test-key")

    assert client._create_args.get("model") == DEFAULT_MODEL


def test_select_response_model_routes_empty_plan_to_cheap_tier() -> None:
    assert select_response_model(Plan()) == CHEAP_MODEL


def test_select_response_model_routes_single_step_to_cheap_tier() -> None:
    plan = Plan(steps=[Step(tool="knowledge_search", search_query="incoterms")])

    assert select_response_model(plan) == CHEAP_MODEL


def test_select_response_model_routes_multi_step_to_strong_tier() -> None:
    plan = Plan(
        steps=[
            Step(tool="tracking_lookup", tracking_number="123"),
            Step(tool="knowledge_search", search_query="rules for {{step_1.destination}}"),
        ]
    )

    assert select_response_model(plan) == STRONG_MODEL


def test_select_response_model_routes_single_sql_step_to_strong_tier() -> None:
    plan = Plan(steps=[Step(tool="sql_query", sql_query="SELECT 1")])

    assert select_response_model(plan) == STRONG_MODEL


def test_select_response_model_routes_single_compliance_step_to_cheap_tier() -> None:
    plan = Plan(steps=[Step(tool="compliance_lookup", category="alcohol", destination="DE")])

    assert select_response_model(plan) == CHEAP_MODEL

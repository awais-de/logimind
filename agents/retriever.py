"""RetrieverAgent: executes the PlannerAgent's plan using DHL tools."""

import logging
import re

from pydantic import BaseModel

from agents.planner import Plan, Step
from agents.tools.compliance_lookup import compliance_lookup
from agents.tools.dhl_search import dhl_knowledge_search
from agents.tools.dhl_tracking_mock import dhl_tracking_mock
from agents.tools.sql_query import run_sql_query
from retrieval.semantic import SearchResult

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{\{step_(\d+)\.(\w+)\}\}")

_StepResult = dict | list[SearchResult] | list[dict]


class RetrievalResult(BaseModel):
    """Everything RetrieverAgent gathered for a Plan.

    Attributes:
        search_results: Chunks found via dhl_knowledge_search, across all
            "knowledge_search" steps in the plan.
        tracking_info: Mock tracking status from the plan's
            "tracking_lookup" step. If a plan has more than one, the last
            one's result wins.
        compliance_results: Matched rules from all "compliance_lookup"
            steps in the plan. A step whose category/destination matched
            no rule contributes nothing here.
        sql_results: Rows returned by all "sql_query" steps in the plan.
    """

    search_results: list[SearchResult] = []
    tracking_info: dict | None = None
    compliance_results: list[dict] = []
    sql_results: list[dict] = []


def _resolve(value: str | None, step_results: list[_StepResult]) -> str | None:
    """Substitute `{{step_N.field}}` placeholders in a step field.

    Args:
        value: The field's raw value, possibly containing placeholders
            referencing an earlier, already-executed step's result.
        step_results: Results of steps executed so far, in order.

    Returns:
        value with any placeholders replaced by the referenced step's
        field value, or None if value is None.

    Raises:
        ValueError: A placeholder references a step that hasn't run yet,
            or a field its result doesn't have. Only steps with a single
            dict result ("tracking_lookup", "compliance_lookup") can be
            referenced this way -- "knowledge_search" and "sql_query"
            steps return a list of results, not one addressable value.
    """
    if value is None:
        return None

    def replace(match: re.Match) -> str:
        step_number, field = match.group(1), match.group(2)
        index = int(step_number) - 1
        if index < 0 or index >= len(step_results):
            raise ValueError(f"Step references step_{step_number}, which hasn't run yet")
        result = step_results[index]
        if not isinstance(result, dict) or field not in result:
            raise ValueError(f"Step references {{{{step_{step_number}.{field}}}}}, but step_{step_number}'s result has no '{field}' field")
        return str(result[field])

    return _PLACEHOLDER_RE.sub(replace, value)


def _execute_step(step: Step, step_results: list[_StepResult]) -> _StepResult:
    """Run one step's tool call, resolving placeholders against prior results."""
    if step.tool == "knowledge_search":
        query = _resolve(step.search_query, step_results)
        return dhl_knowledge_search(query) if query else []

    if step.tool == "tracking_lookup":
        tracking_number = _resolve(step.tracking_number, step_results)
        return dhl_tracking_mock(tracking_number) if tracking_number else {}

    if step.tool == "compliance_lookup":
        category = _resolve(step.category, step_results)
        destination = _resolve(step.destination, step_results)
        if not category or not destination:
            return {}
        return compliance_lookup(category, destination) or {}

    sql_query = _resolve(step.sql_query, step_results)
    return run_sql_query(sql_query) if sql_query else []


def run_retriever(plan: Plan) -> RetrievalResult:
    """Execute a PlannerAgent Plan by calling the tools its steps specify.

    This deliberately isn't an LLM-backed AutoGen agent: PlannerAgent has
    already decided what's needed, so executing that decision is a plain,
    deterministic dispatch rather than something requiring another model
    call. Steps run in order so a later step can reference an earlier
    step's result via a `{{step_N.field}}` placeholder.

    Args:
        plan: The plan produced by PlannerAgent.

    Returns:
        Whatever knowledge-search results, tracking info, compliance
        rules, and/or SQL rows the plan's steps called for. All fields
        are empty/None if the plan had no steps.
    """
    search_results: list[SearchResult] = []
    tracking_info: dict | None = None
    compliance_results: list[dict] = []
    sql_results: list[dict] = []
    step_results: list[_StepResult] = []

    for step in plan.steps:
        result = _execute_step(step, step_results)
        step_results.append(result)

        if step.tool == "knowledge_search":
            search_results.extend(result)
        elif step.tool == "tracking_lookup":
            if result:
                tracking_info = result
        elif step.tool == "compliance_lookup":
            if result:
                compliance_results.append(result)
        else:
            sql_results.extend(result)

    return RetrievalResult(
        search_results=search_results,
        tracking_info=tracking_info,
        compliance_results=compliance_results,
        sql_results=sql_results,
    )

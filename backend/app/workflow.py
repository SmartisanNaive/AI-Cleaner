from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from .database import get_history, insert_history
from .diffing import build_diff, length_warnings
from .nlp.pipeline import choose_nlp_style, rewrite_with_nlp_style
from .prompt_service import build_messages, extract_rewritten_text
from .providers import get_provider
from .providers.base import LLMProvider
from .schemas import RewriteRequest, RewriteResponse
from .settings_service import RuntimeSettings, load_settings


class WorkflowState(TypedDict, total=False):
    request: RewriteRequest
    settings: RuntimeSettings
    provider: LLMProvider
    provider_name: str
    model: str
    original_text: str
    current_text: str
    raw_output: str
    messages: list[dict[str, str]]
    warnings: list[str]
    iteration: int
    evaluator_feedback: str | None
    nlp_applied: bool
    nlp_style: str | None
    diff: list[dict[str, str]]
    record_id: int
    created_at: str


def _resolve_runtime(request: RewriteRequest) -> tuple[RuntimeSettings, LLMProvider, str, str]:
    settings = load_settings()
    provider_name = request.provider or settings.provider
    model = request.model or settings.model_for(provider_name)
    provider = get_provider(settings, provider_name=provider_name, model=model)
    return settings, provider, provider_name, model


async def validate_length_node(state: WorkflowState) -> WorkflowState:
    request = state["request"]
    settings, provider, provider_name, model = _resolve_runtime(request)
    warnings = length_warnings(request.text)
    return {
        **state,
        "settings": settings,
        "provider": provider,
        "provider_name": provider_name,
        "model": model,
        "original_text": request.text,
        "current_text": request.text,
        "warnings": warnings,
        "iteration": 1,
        "nlp_applied": False,
        "nlp_style": None,
    }


async def select_prompt_node(state: WorkflowState) -> WorkflowState:
    request = state["request"]
    messages = build_messages(
        request.platform,
        state["current_text"],
        state.get("evaluator_feedback"),
    )
    return {**state, "messages": messages}


async def llm_rewrite_node(state: WorkflowState) -> WorkflowState:
    raw = await state["provider"].complete(state["messages"])
    rewritten = extract_rewritten_text(state["request"].platform, raw)
    return {**state, "raw_output": raw, "current_text": rewritten}


async def evaluate_iterate_node(state: WorkflowState) -> WorkflowState:
    request = state["request"]
    provider = state["provider"]
    current = state["current_text"]
    raw_output = state["raw_output"]
    feedback: str | None = None

    for iteration in range(2, request.iterations + 1):
        eval_messages = [
            {
                "role": "system",
                "content": (
                    "你是文本改写质量评估 agent。只输出简短可执行的修订意见，"
                    "不得新增事实，不得要求改变原文核心信息。"
                ),
            },
            {
                "role": "user",
                "content": f"原文：\n{request.text}\n\n当前改写：\n{current}",
            },
        ]
        feedback = await provider.complete(eval_messages)
        messages = build_messages(request.platform, current, feedback)
        raw_output = await provider.complete(messages)
        current = extract_rewritten_text(request.platform, raw_output)

    return {
        **state,
        "iteration": request.iterations,
        "evaluator_feedback": feedback,
        "raw_output": raw_output,
        "current_text": current,
    }


async def optional_nlp_classify_node(state: WorkflowState) -> WorkflowState:
    request = state["request"]
    settings = state["settings"]
    enabled = settings.nlp_enabled if request.nlp_enabled is None else request.nlp_enabled
    if not enabled:
        return {**state, "nlp_applied": False, "nlp_style": None}

    mode = request.nlp_mode or settings.nlp_mode
    style = request.nlp_style or settings.nlp_style
    style = await choose_nlp_style(state["current_text"], mode, style, state["provider"])
    return {**state, "nlp_applied": True, "nlp_style": style}


async def optional_nlp_rewrite_node(state: WorkflowState) -> WorkflowState:
    if not state.get("nlp_applied"):
        return state
    style = state.get("nlp_style") or "academic"
    request = state["request"]
    rewritten = await rewrite_with_nlp_style(
        state["current_text"],
        style,
        aggressive=request.nlp_aggressive,
        best_of_n=request.nlp_best_of_n,
        seed=request.nlp_seed,
    )
    return {**state, "current_text": rewritten}


async def build_diff_node(state: WorkflowState) -> WorkflowState:
    return {**state, "diff": build_diff(state["original_text"], state["current_text"])}


async def persist_record_node(state: WorkflowState) -> WorkflowState:
    created_at = datetime.now(timezone.utc).isoformat()
    record_id = insert_history(
        {
            "original_text": state["original_text"],
            "raw_output": state["raw_output"],
            "rewritten_text": state["current_text"],
            "platform": state["request"].platform,
            "provider": state["provider_name"],
            "model": state["model"],
            "iterations": state["request"].iterations,
            "warnings": state["warnings"],
            "nlp_applied": bool(state.get("nlp_applied")),
            "nlp_style": state.get("nlp_style"),
            "diff": state["diff"],
            "created_at": created_at,
        }
    )
    return {**state, "record_id": record_id, "created_at": created_at}


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("validate_length", validate_length_node)
    graph.add_node("select_prompt", select_prompt_node)
    graph.add_node("llm_rewrite", llm_rewrite_node)
    graph.add_node("evaluate_iterate", evaluate_iterate_node)
    graph.add_node("optional_nlp_classify", optional_nlp_classify_node)
    graph.add_node("optional_nlp_rewrite", optional_nlp_rewrite_node)
    graph.add_node("build_diff", build_diff_node)
    graph.add_node("persist_record", persist_record_node)
    graph.set_entry_point("validate_length")
    graph.add_edge("validate_length", "select_prompt")
    graph.add_edge("select_prompt", "llm_rewrite")
    graph.add_edge("llm_rewrite", "evaluate_iterate")
    graph.add_edge("evaluate_iterate", "optional_nlp_classify")
    graph.add_edge("optional_nlp_classify", "optional_nlp_rewrite")
    graph.add_edge("optional_nlp_rewrite", "build_diff")
    graph.add_edge("build_diff", "persist_record")
    graph.add_edge("persist_record", END)
    return graph.compile()


async def run_rewrite(request: RewriteRequest) -> RewriteResponse:
    final_state: WorkflowState = await build_graph().ainvoke({"request": request})
    row = get_history(final_state["record_id"])
    if row is None:
        raise RuntimeError("History record was not persisted.")
    return history_row_to_response(row)


def history_row_to_response(row: Any) -> RewriteResponse:
    return RewriteResponse(
        id=row["id"],
        original_text=row["original_text"],
        rewritten_text=row["rewritten_text"],
        raw_output=row["raw_output"],
        platform=row["platform"],
        provider=row["provider"],
        model=row["model"],
        iterations=row["iterations"],
        warnings=json.loads(row["warnings_json"]),
        nlp_applied=bool(row["nlp_applied"]),
        nlp_style=row["nlp_style"],
        diff=json.loads(row["diff_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def route_after_nlp_classify(state: WorkflowState) -> Literal["optional_nlp_rewrite", "build_diff"]:
    return "optional_nlp_rewrite" if state.get("nlp_applied") else "build_diff"

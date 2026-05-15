from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from .aigc_detector import detect_aigc_risk, format_aigc_report_for_prompt
from .diffing import build_diff, length_warnings
from .nlp.pipeline import choose_nlp_style, rewrite_with_nlp_style
from .prompt_service import build_messages, extract_rewritten_text
from .providers import get_provider
from .providers.base import LLMProvider
from .response_utils import build_rewrite_response
from .runtime_config import resolve_provider_config
from .schemas import RewriteRequest, RewriteResponse


class WorkflowState(TypedDict, total=False):
    request: RewriteRequest
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


def _resolve_runtime(request: RewriteRequest) -> tuple[LLMProvider, str, str]:
    config = resolve_provider_config(
        provider_name=request.provider,
        model=request.model,
        base_url=request.base_url,
        api_key=request.api_key,
    )
    provider = get_provider(config)
    return provider, config.provider, config.model


async def validate_length_node(state: WorkflowState) -> WorkflowState:
    request = state["request"]
    provider, provider_name, model = _resolve_runtime(request)
    warnings = length_warnings(request.text)
    return {
        **state,
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
        detection_report = detect_aigc_risk(current)
        detection_prompt = format_aigc_report_for_prompt(detection_report)
        eval_messages = [
            {
                "role": "system",
                "content": (
                    "你是中文论文 AIGC 检测风险评估 agent。"
                    "你的任务不是润色文章，而是结合本地检测器结果，找出当前改写文本中仍可能被检测为 AI 的语言特征。"
                    "请只输出简短、可执行的修订意见，禁止输出改写后的正文。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请结合本地 humanize-chinese 检测结果，对下面文本做 AIGC 检测风险评估，并给出下一轮修订方向。\n\n"
                    "本地检测结果：\n"
                    f"{detection_prompt}\n\n"
                    "你需要重点检查：\n"
                    "1. 是否存在模板化连接词或总结腔，如“首先、其次、最后、综上、显著意义、现实而迫切”等。\n"
                    "2. 是否存在过于工整的并列结构、排比式表达或均匀句长。\n"
                    "3. 是否存在抽象名词连续堆叠、动词弱化、表达过度凝练的问题。\n"
                    "4. 是否缺少人工写作中常见的节奏变化、解释性转折和自然语序。\n"
                    "5. 哪些句子应拆分、调序、换成更自然的学术转述。\n\n"
                    "输出要求：\n"
                    "- 最多 6 条。\n"
                    "- 每条必须是具体可执行的修改建议。\n"
                    "- 优先引用本地检测结果中的风险词、风险句或统计特征。\n"
                    "- 不得改变事实、数据、术语和结论。\n\n"
                    f"原文：\n{request.text}\n\n当前改写：\n{current}"
                ),
            },
        ]
        feedback = await provider.complete(eval_messages)
        messages = build_messages(request.platform, current, feedback, iteration=iteration)
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
    if not request.nlp_enabled:
        return {**state, "nlp_applied": False, "nlp_style": None}

    mode = request.nlp_mode or "manual"
    style = request.nlp_style or "academic"
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


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("validate_length", validate_length_node)
    graph.add_node("select_prompt", select_prompt_node)
    graph.add_node("llm_rewrite", llm_rewrite_node)
    graph.add_node("evaluate_iterate", evaluate_iterate_node)
    graph.add_node("optional_nlp_classify", optional_nlp_classify_node)
    graph.add_node("optional_nlp_rewrite", optional_nlp_rewrite_node)
    graph.add_node("build_diff", build_diff_node)
    graph.set_entry_point("validate_length")
    graph.add_edge("validate_length", "select_prompt")
    graph.add_edge("select_prompt", "llm_rewrite")
    graph.add_edge("llm_rewrite", "evaluate_iterate")
    graph.add_edge("evaluate_iterate", "optional_nlp_classify")
    graph.add_edge("optional_nlp_classify", "optional_nlp_rewrite")
    graph.add_edge("optional_nlp_rewrite", "build_diff")
    graph.add_edge("build_diff", END)
    return graph.compile()


async def run_rewrite(request: RewriteRequest) -> RewriteResponse:
    final_state: WorkflowState = await build_graph().ainvoke({"request": request})
    return build_rewrite_response(
        original_text=final_state["original_text"],
        rewritten_text=final_state["current_text"],
        raw_output=final_state["raw_output"],
        platform=request.platform,
        provider=final_state["provider_name"],
        model=final_state["model"],
        iterations=request.iterations,
        warnings=final_state["warnings"],
        nlp_applied=bool(final_state.get("nlp_applied")),
        nlp_style=final_state.get("nlp_style"),
        diff=final_state["diff"],
    )


def route_after_nlp_classify(state: WorkflowState) -> Literal["optional_nlp_rewrite", "build_diff"]:
    return "optional_nlp_rewrite" if state.get("nlp_applied") else "build_diff"

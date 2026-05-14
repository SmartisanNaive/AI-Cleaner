from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HUMANIZE_SCRIPTS_DIR = Path(__file__).resolve().parent / "nlp" / "humanize_chinese" / "scripts"
if str(_HUMANIZE_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_HUMANIZE_SCRIPTS_DIR))

from .nlp.humanize_chinese.scripts.detect_cn import (  # noqa: E402
    CATEGORY_NAMES,
    analyze_sentences,
    calculate_score,
    detect_patterns,
    score_to_level,
)


@dataclass(frozen=True)
class AigcDetectionReport:
    score: int
    level: str
    total_issues: int
    metrics: dict[str, Any]
    issue_summaries: list[str]
    risky_sentences: list[str]


def _severity_rank(severity: str) -> int:
    return {
        "critical": 0,
        "high": 1,
        "statistical": 2,
        "medium": 3,
        "style": 4,
    }.get(severity, 9)


def _iter_issue_rows(issues: dict[str, list[dict[str, Any]]]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    for category, items in issues.items():
        for item in items:
            yield category, str(item.get("severity", "medium")), item


def detect_aigc_risk(text: str, *, max_issues: int = 10, max_sentences: int = 5) -> AigcDetectionReport:
    """Run the bundled humanize-chinese detector and return compact LLM-ready facts.

    The score is heuristic: higher means more AI-like according to local rule/statistical
    signals. It is used as guidance for iterative rewriting, not as a definitive verdict.
    """
    issues, metrics = detect_patterns(text)
    score = calculate_score(issues, metrics)
    level = score_to_level(score)
    total_issues = sum(len(items) for items in issues.values())

    rows = sorted(
        _iter_issue_rows(issues),
        key=lambda row: (_severity_rank(row[1]), -int(row[2].get("count", 1))),
    )

    summaries: list[str] = []
    for category, severity, item in rows[:max_issues]:
        _, name = CATEGORY_NAMES.get(category, ("", category))
        text_value = str(item.get("text", "")).strip()
        count = int(item.get("count", 1))
        count_part = f"，出现 {count} 次" if count > 1 else ""
        summaries.append(f"{name}｜{severity}｜{text_value}{count_part}")

    risky_sentences: list[str] = []
    for row in analyze_sentences(text, top_n=max_sentences):
        sentence = str(row.get("sentence", "")).strip()
        reasons = "、".join(str(reason) for reason in row.get("reasons", [])[:3])
        score_part = row.get("score", 0)
        if sentence:
            risky_sentences.append(f"[{score_part}分] {sentence}" + (f"（原因：{reasons}）" if reasons else ""))

    return AigcDetectionReport(
        score=score,
        level=level,
        total_issues=total_issues,
        metrics=metrics,
        issue_summaries=summaries,
        risky_sentences=risky_sentences,
    )


def format_aigc_report_for_prompt(report: AigcDetectionReport) -> str:
    """Compress detector output into actionable prompt context."""
    metric_parts: list[str] = []
    for key, label in [
        ("perplexity", "困惑度"),
        ("burstiness", "突发度"),
        ("entropy_cv", "段落熵CV"),
        ("entropy", "字符熵"),
        ("emotional_density", "情感/个人表达密度"),
    ]:
        value = report.metrics.get(key)
        if isinstance(value, (int, float)):
            metric_parts.append(f"{label}={value:.3g}")

    lines = [
        f"本地 humanize-chinese 检测分数：{report.score}/100，风险等级：{report.level}，问题数：{report.total_issues}。",
    ]
    if metric_parts:
        lines.append("统计特征：" + "；".join(metric_parts) + "。")

    if report.issue_summaries:
        lines.append("主要风险点：")
        lines.extend(f"- {item}" for item in report.issue_summaries)
    else:
        lines.append("主要风险点：未命中明显规则风险，下一轮应优先做细微节奏和语序调整，避免过度重写。")

    if report.risky_sentences:
        lines.append("高风险句子/片段：")
        lines.extend(f"- {item}" for item in report.risky_sentences)

    lines.append(
        "请把这些检测结果转化为具体修订动作：替换高风险套话，打散过工整结构，增加句长变化和自然停顿，降低抽象名词堆叠；同时保持事实、术语和结论不变。"
    )
    return "\n".join(lines)

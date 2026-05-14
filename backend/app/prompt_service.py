from __future__ import annotations

import re

from .constants import PROMPTS_DIR, SUPPORTED_PLATFORMS


def get_prompt(platform: str) -> str:
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}")
    path = PROMPTS_DIR / f"{platform}.md"
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def build_messages(
    platform: str,
    text: str,
    evaluator_feedback: str | None = None,
    iteration: int | None = None,
) -> list[dict[str, str]]:
    prompt = get_prompt(platform)
    suffix = ""
    if evaluator_feedback:
        iteration_line = f"这是第 {iteration} 轮修订。\n" if iteration else ""
        suffix = (
            "\n\n迭代修订任务：\n"
            f"{iteration_line}"
            "上一轮文本已经完成基础改写，但仍需要继续降低 AIGC 检测风险。\n"
            "请优先根据下面的评估意见修订，而不是重新写一篇新文章。\n\n"
            "上一轮检测风险评估意见：\n"
            f"{evaluator_feedback}\n\n"
            "本轮修订要求：\n"
            "1. 保持原文事实、数据、术语、研究对象和结论不变，不新增未经原文支持的信息。\n"
            "2. 重点降低机器感：减少模板化连接词、过度工整的并列结构、连续抽象名词堆叠和过强总结腔。\n"
            "3. 调整句子节奏：适当拆分过长句，穿插中等长度句；避免每句都呈现同一种逻辑结构。\n"
            "4. 让表达更像人工论文修改：保留学术严谨性，但使用更自然的转述、解释性补充和轻微语序变化。\n"
            "5. 不要为了降重而堆砌口水词，不要明显扩写，整体字数尽量控制在上一轮文本的 ±10% 内。\n"
            "6. 只输出修订后的正文，不输出分析、标题、说明或项目符号。"
        )
    return [
        {"role": "system", "content": prompt + suffix},
        {"role": "user", "content": f"原文：\n{text}"},
    ]


ARTICLE_START_MARKERS = [
    "重写后：",
    "重写后",
    "重写后的文本：",
    "改写后：",
    "改写后",
    "重写后的文章：",
    "改写后的文本：",
    "改写后的文章：",
    "优化后：",
    "优化后",
    "优化后的文本：",
    "优化后的文章：",
    "修改后：",
    "修改后",
    "修改后的文本：",
    "修改后的文章：",
    "最终文本：",
    "最终版本：",
    "正文：",
]

ARTICLE_END_MARKERS = [
    "修改细节说明",
    "修改说明",
    "改写说明",
    "优化说明",
    "调整说明",
    "对照您的要求",
    "原文 AI 特征分析",
    "核心优化策略",
    "优化亮点说明",
]


def _strip_markdown_heading(line: str) -> str:
    return re.sub(r"^\s{0,3}#{1,6}\s*", "", line).strip()


def _normalize_marker_text(line: str) -> str:
    line = _strip_markdown_heading(line)
    line = line.strip().strip("*-_ `\t")
    line = re.sub(r"^\d+[.、．]\s*", "", line)
    return line.strip().strip("：:").strip()


def _looks_like_meta_intro(text: str) -> bool:
    compact = text.strip()
    if len(compact) > 120:
        return False
    meta_words = ["为您提供", "以下是", "下面是", "版本", "重写", "改写", "优化", "符合", "要求"]
    return any(word in compact for word in meta_words)


def clean_rewritten_text(raw_output: str) -> str:
    """Keep only the final rewritten article body from chatty model output."""
    text = raw_output.strip()
    if not text:
        return ""

    # Prefer explicit article-start markers and discard anything before them.
    best_start: int | None = None
    for marker in ARTICLE_START_MARKERS:
        patterns = [
            rf"(?:\*\*)?{re.escape(marker)}(?:\*\*)?",
            rf"(?:\*\*)?{re.escape(marker.rstrip('：'))}\s*[:：](?:\*\*)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match and (best_start is None or match.start() < best_start):
                best_start = match.end()
    if best_start is not None:
        text = text[best_start:].strip()

    # Remove trailing explanation sections such as "修改细节说明".
    earliest_end: int | None = None
    for marker in ARTICLE_END_MARKERS:
        pattern = rf"(?:^|\n)\s*(?:[-*_]{{3,}}\s*\n)?\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(?:\d+[.、．]\s*)?{re.escape(marker)}"
        match = re.search(pattern, text, flags=re.I)
        if match and (earliest_end is None or match.start() < earliest_end):
            earliest_end = match.start()
    if earliest_end is not None:
        text = text[:earliest_end].strip()

    lines = text.splitlines()

    # Drop leading meta sentence if the model says "这里为您提供..." before body.
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and _looks_like_meta_intro(lines[0]):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    # Drop standalone markdown separators left around the body.
    cleaned_lines = []
    for line in lines:
        if re.fullmatch(r"\s*[-*_]{3,}\s*", line):
            continue
        normalized = _normalize_marker_text(line)
        if normalized in {m.rstrip("：") for m in ARTICLE_START_MARKERS}:
            continue
        cleaned_lines.append(line.rstrip())

    text = "\n".join(cleaned_lines).strip()

    # If a model still leaks a numbered explanation block after the article, cut it.
    leak_patterns = [
        r"\n\s*1[.、．]\s*\*\*?剔除",
        r"\n\s*1[.、．]\s*\*\*?去除",
        r"\n\s*1[.、．]\s*\*\*?修改",
        r"\n\s*1[.、．]\s*\*\*?优化",
    ]
    for pattern in leak_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[: match.start()].strip()
            break

    return text.strip()


def extract_rewritten_text(platform: str, raw_output: str) -> str:
    return clean_rewritten_text(raw_output)

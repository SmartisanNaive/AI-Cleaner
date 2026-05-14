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


def build_messages(platform: str, text: str, evaluator_feedback: str | None = None) -> list[dict[str, str]]:
    prompt = get_prompt(platform)
    suffix = ""
    if evaluator_feedback:
        suffix = f"\n\n上一轮评估意见：\n{evaluator_feedback}\n\n请在不改变事实的前提下继续修订。"
    return [
        {"role": "system", "content": prompt + suffix},
        {"role": "user", "content": f"原文：\n{text}"},
    ]


def extract_rewritten_text(platform: str, raw_output: str) -> str:
    if platform != "paperyy":
        return raw_output.strip()
    match = re.search(r"(?:\*\*)?优化后的文章：(?:\*\*)?\s*(.*)", raw_output, flags=re.S)
    if match:
        return match.group(1).strip()
    marker = "优化后的文章："
    if marker in raw_output:
        return raw_output.split(marker, 1)[1].strip()
    alt = "4. **优化后的文章：**"
    if alt in raw_output:
        return raw_output.split(alt, 1)[1].strip()
    return raw_output.strip()

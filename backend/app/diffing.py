from __future__ import annotations

from difflib import SequenceMatcher


def build_diff(original: str, revised: str) -> list[dict[str, str]]:
    spans: list[dict[str, str]] = []
    matcher = SequenceMatcher(None, original, revised, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        spans.append(
            {
                "kind": tag if tag != "replace" else "replace",
                "original": original[i1:i2],
                "revised": revised[j1:j2],
            }
        )
    return spans


def char_count(text: str) -> int:
    return len([c for c in text if c.strip()])


def length_warnings(text: str) -> list[str]:
    count = char_count(text)
    if count < 300:
        return [f"当前文本约 {count} 字，建议输入 300-1200 字，否则效果可能不稳定。"]
    if count > 1200:
        return [f"当前文本约 {count} 字，建议输入 300-1200 字，否则效果可能不稳定。"]
    return []


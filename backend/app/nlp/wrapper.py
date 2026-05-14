from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Literal

VENDOR_DIR = Path(__file__).resolve().parent / "humanize_chinese" / "scripts"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

NlpStyle = Literal["academic", "general", "long_blog"]


def classify_locally(text: str) -> NlpStyle:
    academic_markers = ("研究", "本文", "本研究", "文献", "理论", "实证", "样本", "模型")
    social_markers = ("我", "你", "大家", "分享", "评论", "点赞")
    if sum(text.count(marker) for marker in academic_markers) >= 2:
        return "academic"
    if len(text) > 1200 or sum(text.count(marker) for marker in social_markers) >= 2:
        return "long_blog"
    return "general"


def parse_llm_classification(raw: str) -> NlpStyle | None:
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    style = data.get("style") or data.get("nlp_style") or data.get("category")
    if style in {"academic", "general", "long_blog"}:
        return style
    return None


def apply_nlp(
    text: str,
    style: NlpStyle,
    aggressive: bool = False,
    best_of_n: int = 10,
    seed: int | None = None,
) -> str:
    if style == "academic":
        from academic_cn import humanize_academic

        return humanize_academic(
            text,
            aggressive=aggressive,
            seed=seed,
            best_of_n=best_of_n,
        )
    from humanize_cn import humanize

    if style == "long_blog":
        return humanize(
            text,
            scene="general",
            aggressive=aggressive,
            seed=seed,
            best_of_n=best_of_n,
            score_mode="fused",
        )
    return humanize(
        text,
        scene="general",
        aggressive=aggressive,
        seed=seed,
        best_of_n=best_of_n,
    )

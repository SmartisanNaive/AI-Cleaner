from __future__ import annotations

from datetime import datetime, timezone

from backend.app.diffing import build_diff, length_warnings
from backend.app.nlp.pipeline import choose_nlp_style
from backend.app.nlp.wrapper import classify_locally, parse_llm_classification
from backend.app.prompt_service import extract_rewritten_text, get_prompt
from backend.app.schemas import RewriteResponse
from backend.app.security import decrypt_secret, encrypt_secret


def test_encrypt_secret_roundtrip():
    encrypted = encrypt_secret("sk-test")
    assert encrypted and encrypted != "sk-test"
    assert decrypt_secret(encrypted) == "sk-test"


def test_length_warnings():
    assert length_warnings("短文本")
    assert length_warnings("字" * 1300)
    assert length_warnings("字" * 400) == []


def test_diff_marks_replace():
    spans = build_diff("人工智能发展", "人工智能稳步发展")
    assert any(span["kind"] in {"insert", "replace"} for span in spans)


def test_paperyy_extracts_final_article():
    raw = "1. **原文 AI 特征分析：**\n略\n4. **优化后的文章：**\n这里是正文"
    assert extract_rewritten_text("paperyy", raw) == "这里是正文"


def test_prompts_are_loadable():
    assert "论文" in get_prompt("weipu")
    assert "AI 文章润色师" in get_prompt("paperyy")


def test_nlp_classification_helpers():
    assert parse_llm_classification('{"style":"academic"}') == "academic"
    assert classify_locally("本文基于已有研究构建理论模型，并结合样本进行实证分析。") == "academic"


async def test_nlp_pipeline_can_classify_without_llm():
    text = "本文基于已有研究构建理论模型，并结合样本进行实证分析。"
    assert await choose_nlp_style(text, "auto", None) == "academic"
    assert await choose_nlp_style(text, "manual", "unknown") == "academic"


def test_rewrite_response_accepts_local_nlp_provider():
    response = RewriteResponse(
        id=1,
        original_text="原文",
        rewritten_text="改写",
        raw_output="改写",
        platform="weipu",
        provider="local",
        model="humanize-chinese",
        iterations=0,
        warnings=[],
        nlp_applied=True,
        nlp_style="general",
        diff=[],
        created_at=datetime.now(timezone.utc),
    )
    assert response.provider == "local"

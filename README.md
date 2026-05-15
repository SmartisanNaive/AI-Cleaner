# AI-Cleaner

本地单机文本改写与 Agent 架构测试项目。项目使用 FastAPI、LangGraph、OpenAI SDK、Anthropic SDK、React/Vite 和 `@chenglou/pretext`，并内嵌 `humanize-chinese` 作为离线 NLP 强力模式参考实现。

> 目标是验证本地 Agent 工作流、Provider 适配、文本差异可视化和离线 NLP 改写算法。请勿将本项目用于学术不端、欺骗性提交或绕过平台规则。

## 运行

```bash
uv sync
uv run uvicorn backend.app.main:app --reload --port 8000
```

另开一个终端：

```bash
cd frontend
pnpm install
pnpm dev
```

前端默认访问 `http://127.0.0.1:5173`，后端默认 `http://127.0.0.1:8000`。

## 学术论文 AIGC 降痕

- 普通“改写”按钮仍按 LLM 工作流执行；开启“改写后追加学术降痕”时，会在模型改写后追加 `academic_cn.py` 学术论文降痕处理。
- “仅学术降痕”按钮只调用本地 `humanize-chinese` 的 academic 管线，不走 OpenAI/Anthropic SDK，并用打字机效果流式输出结果。
- 默认类型为 `academic`，默认候选数 `best_of_n=10`，对齐上游 `academic_cn.py` / `academic.md` 的学术论文主路径。
- 后端独立接口为 `POST /api/nlp`，流式接口为 `POST /api/nlp/stream`，请求字段包括 `text`、`platform`、`nlp_mode`、`nlp_style`、`best_of_n`、`seed`、`aggressive`。
- LLM 工作流的追加 NLP 支持同样参数：`nlp_best_of_n`、`nlp_seed`、`nlp_aggressive`。

## 环境变量

API Key 优先读取环境变量；设置页保存的 Key 只作为本机加密 fallback。

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4
OPENAI_BASE_URL=https://api.openai.com/v1
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-4-6-sonnet
ANTHROPIC_BASE_URL=https://api.anthropic.com
AI_CLEANER_SECRET=optional-local-fernet-secret
```

## 结构

- `backend/app`: FastAPI 后端、LangGraph 工作流、SQLite 持久化、Provider 适配器。
- `backend/app/prompts`: 用户提供的平台 Prompt，按文件原样加载。
- `backend/app/nlp/humanize_chinese`: `humanize-chinese` 上游代码与许可证。
- `frontend/src`: React 主界面、设置页、Pretext canvas 背景和对比视图。

## 测试

```bash
uv run pytest
cd frontend
pnpm test
```

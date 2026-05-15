# **AI Cleaner**

*404: AI signature not found*

## **什么是 AI Cleaner？**

AI Cleaner 是一个基于 NLP 和 LangGraph 架构降低 AI 率的 Agent 系统，经过多轮实测，能有效在各大中文 AI 率检测平台展现出比较优秀的降 AI 率效果。

> 本项目用于神经语言程序研究、Agent 工作流设计验证和人机协同实验。请勿将其用于学术不端、伪造原创或规避平台规则。
> 

### **核心能力**

- 低性能要求：在独立 NLP 算法下，使用零 token 直接对文本进行改写；
- 多轮改写：通过 LangGraph 状态机驱动改写 Agent 与评估 Agent 迭代协作，每轮由本地 AIGC 检测器产出风险报告，指导下一轮修订方向；
- 前端优雅：支持流式输出、差异对比、历史记录和设置页调试
- 项目透明：完整公开 LangGraph 工作流定义、Prompt 构造、NLP 管线、AIGC 检测逻辑及安全边界；
- 效果绝佳：LLM 改写 + 本地 NLP 后处理双通道串联，前者负责语义重构与风格变换，后者以规则+统计方式做句式打散、节奏扰动和去模板化，实测在各大中文 AIGC 检测平台均有稳定降 AI 率表现。

### **你可以用 AI Cleaner 做什么**

把一段中文文本丢进去，它会帮你改写成"**人看着觉得像 AI 写的，但是算法检测不出来**“的版本。

- 降低学术论文段落的 AI 痕迹；
- 消除 AIGC 文本的"总结腔"和"模板感"；
- 也许对初学者来说，是一个很棒的 LangGraph 案例，可以帮助你了解如何构建属于自己的 Agent 项目；

### **为什么 AI Cleaner 能实现降 AIGC 检测**

AI Cleaner 的每一层改写机制都根据主流检测器的判别维度进行优化，主流中文 AIGC 检测器（知x、维x、x方等）是在多个维度捕捉**机器写作的统计指纹**，其中核心判别维度如下：

| **检测维度** | **技术含义** | **典型阈值（HC3 校准）** |
| --- | --- | --- |
| **困惑度** (Perplexity) | 文本对语言模型"过于可预测"→ AI 特征 | 低于人类文本分布均值 |
| **句长变异系数** (Sentence Length CV) | AI 句子长度分布异常均匀 | Cohen's d = 1.22，最强单维信号 |
| **短句缺失率** | AI 几乎不写 10 字以内的短句 | 短句占比 < 8% 触发 |
| **过渡词密度** | "然而/此外/综上所述"等连接词过度使用 | > 8次/千字触发 |
| **字符熵** | 用字过于集中，缺乏多样性 | MATTR < 0.65 触发 |
| **局部曲率** (DetectGPT-lite) | 每处选词都选最高概率续接 | GLTR top-10 占比 > 21% 触发 |
| **突发度** (Burstiness) | 困惑度在全文过于均匀，缺少人类写作的起伏 | 困惑度变异系数过低触发 |
| **模板化结构** | "首先…其次…最后"、"一方面…另一方面"等 | 正则匹配直接命中 |
| **AI 高频词库** | "值得注意的是"、"赋能"、"底层逻辑"等 | 动态词库密度检测 |

简单同义词替换几乎无效。以知* AIGC 检测为例，检测深入到语义向量和写作模式层面：把"人工智能技术在医疗领域的应用日益广泛"改成"AI科技于医学范畴之运用愈发普遍"，AIGC 率仅从 75% 降到 72%，因为信息密度、逻辑结构、语义节奏在向量空间中几乎重合。因此，降 AIGC 必须在**统计特征**和**写作模式**两个层面同时出击。

### **第一层：用大模型先做整体改写，让文本不再像原来的 AI 表达**

第一轮改写主要由 LLM 完成，从表达方式、句子结构和论证顺序上重新组织文本。

项目里内置不同场景下的 Prompt 模板，用来引导模型改写时重点处理这些问题：

- 避免使用过于模板化的连接词，比如“首先、其次、综上所述”；
- 打破过于工整、对称的句式结构；
- 减少空泛的抽象名词和套话式总结；
- 调整原文的论证顺序，让文章的逻辑推进方式发生变化。

如，原文可能是“先讲 A，再推出 B，最后得到 C”，改写后可以变成“先说明 C 为什么重要，再回到 B 和 A 进行解释”，文本在语义结构上和原文拉开距离。

### **第二层：用 Agent 多轮检查和修改，处理高风险句子**

只改一遍（即直接用 Prompt 让不同模型重新写一遍）通常很难把上述所有问题都处理干净。 AI Cleaner 通过 LangGraph 组织多个 Agent 反复协作：

> 改写 → 本地检测 → 找出问题 → 给出修改建议 → 定向改写 → 再次检测
> 

Agent 会先用本地检测器分析当前文本，找出哪些地方仍然比较像 AI 写的，比如：

- 哪些句子的 Perplexity 异常、长度平均，缺少变化；
- 哪些过渡词使用密集；
- 哪些模板化句式没有清理干净。

此部分为局部重写，避免文本在多轮改写后仍然停留在同一种 AI 风格里。

### **第三层：用本地 NLP 规则做后处理**

这是 AI Cleaner 和其他项目最大的区别之一。

在 LLM 完成改写后，系统会通过本地 NLP 管线做进一步处理，不调用大模型，也不消耗 token。事实上，即使 LLM 已经改过一轮，文本里仍然可能保留某种“AI 味”。

## **部署**

### **Docker 部署（推荐）**

#### **前置要求**

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

#### **1. 克隆项目**

```
git clone https://github.com/your-repo/AI-Cleaner.git
cd AI-Cleaner
```

#### **2. 配置环境变量**

复制并编辑 `.env` 文件，Docker compose 会默认读取项目根目录下的 .env。至少需要准备：

```python
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4
OPENAI_BASE_URL=https://api.openai.com/v1
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-4-6-sonnet
ANTHROPIC_BASE_URL=https://api.anthropic.com
AI_CLEANER_SECRET=optional-local-fernet-secret
AI_CLEANER_PORT=8080
```

> 支持所有 OpenAI 兼容的 API（如 Azure OpenAI、DeepSeek、Ollama 等），只需修改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL` 即可。
> 
> 
> 个人体感而言，Deepseek-v4-Flash 模型的效果在这方面不会比御三家的能力差，没必要为了一个简单的降 AI 任务而上 SOTA。
> 

#### **3. 一键启动**

```
docker compose up --build -d
```

首次启动会自动构建镜像（构建前端 + 安装后端依赖），大约需要 1-2 分钟。

启动后访问 [**http://localhost:8000**](http://localhost:8000/) 即可使用。

此处的常用命令：

```python
# 仅构建镜像
docker compose build

# 构建并后台启动前后端
docker compose up --build -d

# 仅启动已有容器
docker compose up -d

# 查看服务状态
docker compose ps

# 查看全部日志
docker compose logs -f

# 仅查看后端日志
docker compose logs -f backend

# 仅查看前端日志
docker compose logs -f frontend

# 进入后端容器
docker compose exec backend sh

# 进入前端容器
docker compose exec frontend sh

# 重启某个服务
docker compose restart backend

# 停止服务但保留容器
docker compose stop

# 停止并删除容器和网络
docker compose down

# 修改代码或依赖后重新构建并启动
docker compose up --build -d
```

此处说明：

- frontend 容器负责托管 Vite 构建后的静态文件，并通过 Nginx 反向代理 /api 到 backend。
- backend 容器运行 FastAPI，容器内监听 8000，默认不直接暴露到宿主机。
- ./data 会挂载到容器内 /app/data，用于持久化本地生成的 app.secret 等数据。
- 如果你想修改对外端口，可以在 .env 里设置 AI_CLEANER_PORT，然后重新执行 docker compose up -d。
- 如果你修改了 Python 或前端依赖，建议重新执行 docker compose up --build -d 触发镜像重建。

#### **4. 自定义端口**

默认映射到宿主机 8000 端口。如果端口冲突，可以在 `.env` 中添加：

```
HOST_PORT=9000
```

#### **5. 数据持久化**

Docker 使用 named volume `ai-cleaner-data` 持久化 SQLite 数据库和密钥，容器重建不会丢失数据。

查看数据卷位置：

```
docker volume inspect ai-cleaner_ai-cleaner-data
```

#### **6. 更新部署**

```
git pull
docker compose up -d --build
```

#### **7. 停止服务**

```
docker compose down
# 如需同时删除数据卷
docker compose down -v
```

### **手动部署（本地）**

#### **后端**

```
# 安装 uv（Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync

# 启动后端服务
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

#### **前端**

```
cd frontend

# 安装 pnpm（如果没有）
npm install -g pnpm

# 安装依赖
pnpm install

# 开发模式（带热更新 + API 代理）
pnpm dev
# 访问 http://localhost:5173

# 或者构建生产版本
pnpm build
# 构建产物在 frontend/dist/，可由 FastAPI 直接 serve
```

## **Cloud 分支说明**

Cloud 版本只用于本项目演示部分的项目信息披露，保证演示的网页端不会收集相关敏感信息，Cloud ****分支下，用户在浏览器中输入的 API key 不会被后端长期保存。后端目前只保留了无状态的接口（例如 `/api/settings/test`、`/api/rewrite` 和 `/api/nlp*` ），API key 只会在用户发起请求时短暂进入后端内存，并被转发给所选择的上游服务，如果用户没有填写 API key，系统才会使用服务端配置的环境变量作为兜底。同时，后端已经对 `/api/*` 响应统一设置了 `Cache-Control: no-store`，并且会对日志中的 `api_key`、`authorization`、`base_url` 等敏感字段做脱敏处理，降低泄露风险。

前端默认“不记住 API key / 不记住历史”。只有当用户主动勾选相关选项，并点击“保存到浏览器”后，信息才会写入本地 `localStorage`。

但仍请注意以下几点：

- Cloud 分支用于云端演示版，后端只做临时转发，不保存用户的 API Key、Base URL、输入正文、输出结果或历史记录。此部分主要是为了让云端演示的项目透明，不建议个人部署，建议团队部署进行二次开发。
- 云端演示版中，设置和历史记录只保存在浏览器本地；服务端不再提供 /api/settings 与 /api/history* 持久化接口。
- 可通过 AI_CLEANER_ALLOWED_ORIGINS 配置允许访问的前端域名，多个域名用英文逗号分隔。

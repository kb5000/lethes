# 本地独立服务部署与持续开发

本指南对应 `main` 主分支。lethes 与原 lethes-observer
现在放在同一仓库，一个独立服务同时提供上下文 API 与观测 Web。
Open WebUI 使用轻量 HTTP Filter 调用它，主模型仍由 Open WebUI 请求。

```text
Open WebUI ── 完整当前历史 ──► lethes :8000 ──► SQLite /data/lethes.sqlite3
           ◄── 组装后的消息 ──          ├── /observer/ 观测 Web
           ── provider usage ─►        └── 可选 embedding :80
```

保留你已有的 Open WebUI Compose。这里的 Compose 负责 lethes、观测 Web 及可选小模型。
默认先用关键词召回和字节估算；匹配的历史真实 usage 始终优先。之后再启用 LiteLLM
计数、语义召回及按真实价格判断经济重建。

## 1. 下载与初始化

需要 Git、Docker、Docker Compose v2。初始化脚本使用 Python 3；宿主机不需要安装
lethes 或推理依赖。命令在仓库根目录、Bash/WSL 或类似终端中运行。

```bash
git clone --branch main --single-branch https://github.com/kb5000/lethes.git
cd lethes
python3 scripts/init_local.py
```

脚本生成 `.env` 中的随机 `LETHES_API_KEY` 和 `deploy/lethes.local.json`，不覆盖已有文件，
不打印密钥。这两份本地配置均被 Git 忽略。没有宿主机 Python 时，可以手动复制
`.env.example` 为 `.env`、`deploy/lethes.example.json` 为 `deploy/lethes.local.json`，
再把 `openssl rand -hex 32` 生成的值填入 `.env` 的 `LETHES_API_KEY`。

该 Key 是本地服务的管理凭证，供 Open WebUI 后端和观测页连接，能读取所有请求观测。
API 拒绝少于 16 字符的 Key。后续重启和迭代保留同一值即可。

## 2. 开发模式：修改源码后容器直接响应

```bash
docker compose -f compose.yaml -f compose.dev.yaml config --quiet
docker compose -f compose.yaml -f compose.dev.yaml up -d --build
docker compose logs -f lethes
```

打开 `http://127.0.0.1:8000/`，输入 `.env` 中的 API Key。页面每两秒读取最近
100 条请求及更新后的 usage。Key 只保留在当前页面内存，刷新页面后重新输入，
不会写入 URL 或浏览器持久存储。

| 修改内容 | 生效方式 |
|---|---|
| `src/lethes/**/*.py` | 源码挂载到容器，Uvicorn 自动重载 |
| `src/lethes/observer/` 中的 HTML、JS、CSS | 静态源码直接挂载，刷新浏览器生效，响应禁用缓存 |
| `deploy/lethes.local.json` | 开发模式自动重载并读取配置 |
| `.env` | 再运行同一条 `up -d` 命令，更新容器环境 |
| `pyproject.toml`、`uv.lock`、Dockerfile | 再运行同一条 `up -d --build` 命令 |
| Open WebUI 的 HTTP Filter 源码 | 在 Open WebUI Functions 中重新粘贴并保存 |

`./src` 挂到 `/app/src`，Python 依赖在 `/opt/lethes-venv`，二者互不遮盖。
默认开启文件轮询，方便 Docker Desktop/WSL 的目录挂载触发重载。
SQLite 位于独立数据卷，重载不会清空原文布局、请求观测和真实计数。
添加依赖后先运行 `uv lock`，提交依赖声明与锁文件，再重建镜像。
镜像使用锁定依赖；启动时直接运行 Uvicorn，不重新安装包。

## 3. 接入已有 Open WebUI Compose

默认网络名为 `lethes-local`。在已有 Open WebUI Compose 中增加外部网络，
保留原来的网络、卷、端口、密钥和模型连接配置：

```yaml
services:
  open-webui:
    # 保留你的其他配置
    networks:
      - default
      - lethes
networks:
  lethes:
    external: true
    name: lethes-local
```

先启动本仓库的 Compose，使网络存在，再在 Open WebUI 的目录执行其原有启动命令。
网络名可通过 `.env` 的 `LETHES_NETWORK` 修改，两边保持一致。
共享网络后，Open WebUI 容器用 `http://lethes:8000` 连接。
容器中的 localhost 指向自身。

已有 Open WebUI 若设置了 HTTP/HTTPS 代理，在其 NO_PROXY/no_proxy 的原有列表中加入
`localhost,127.0.0.1,lethes,embedding`，让本地服务直连；外部模型 API 继续使用原代理。
本仓库 Compose 已为 lethes 加上这些本地地址。

在 Open WebUI 管理后台 Functions 新建函数，粘贴：

```text
src/external/open_webui_service_filter.py
```

该文件只依赖 Open WebUI 通常已有的 httpx 和 pydantic，不要求把 lethes、LiteLLM
或模型权重装进 Open WebUI。激活函数、绑定目标聊天模型，并启用聊天中的开关。

| Filter Valve | 设置 |
|---|---|
| `service_url` | `http://lethes:8000` |
| `service_api_key` | 本仓库 `.env` 的 `LETHES_API_KEY` |
| `timeout_seconds` | 默认 60，覆盖首次检索及计数器初始化 |
| `show_status` | `true`，显示一行 token 状态 |

容量、tokenizer、召回、价格统一在 lethes 服务的配置文件中设置。
独立服务使用带 `service` 的新 Filter；旧 `open_webui_filter.py` 属于进程内模式。
同一聊天只启用一个 lethes Filter，关闭其他历史裁剪及 Open WebUI 自带聊天自动压缩，
让服务拿到完整当前历史。

Open WebUI 必须支持 `request()` Hook。2026-11-06 已检查 v0.13.0 的初始请求和
工具后续请求均调用该 Hook；其他版本请核对对应实现。

## 4. 容量、模型与真实计数

修改 `deploy/lethes.local.json`。`defaults` 是默认配置，`models` 按 Open WebUI
实际请求模型 ID 覆盖，未指定字段递归继承默认值。如果请求 ID 没有匹配项，也会
尝试工作区 `base_model_id`。默认配置已在 `deploy/lethes.example.json` 中给出。

例如，在 `models` 中加入：

```json
"my-workspace-model": {
  "tokenizer": "litellm",
  "tokenizer_model": "gpt-4o",
  "context": {"context_window": 32768}
}
```

两处模型名都是示例，改成真实请求 ID 及其实际基础模型。不要把不同模型随意映射
成 GPT 来绕过报错。已知 LiteLLM 模型可省略 `tokenizer_model`，自动采用工作区基础
模型或请求模型；未知别名会要求明确映射。镜像已包含 LiteLLM 可选依赖。

`context_window` 填实际启用的容量。输入上限为容量减 `reply_reserve` 和
`safety_reserve`；若 Open WebUI 请求更大的回复上限，继续扣除额外预留。
输出 token 不计入输入量。`recent_tokens` 是近期软预算，最低连续轮次由
`minimum_recent_turns` 保护，其余历史按需要召回原文。

同模型、同消息的真实输入 usage 优先于本地估算，重启后仍有效；追加消息采用
实测前缀加增量估算，编辑、分叉和重建后的内容明确标为估算。

### 前缀缓存价格

默认 `pricing: null` 关闭经济重建，容量及必要召回重建仍执行。知道供应商实际
价格后，可在默认配置或某个模型覆盖中填入：

```json
"pricing": {
  "uncached_per_million": 2.5,
  "cached_per_million": 0.25,
  "minimum_cache_tokens": 1024,
  "cache_block_tokens": 128,
  "ttl_seconds": 300
}
```

数值仅演示结构，单位为美元/百万输入 token，改成实际计费和缓存条件。
本地模型没有此类价差时保持 null。算法按持续聊天的缓存周期平均成本判断，
不预测用户还会聊多少轮；实际缓存命中以返回 usage 为准。

## 5. 可选：很小的本地 embedding 服务

Compose 的 semantic profile 提供 BGE-small-zh-v1.5 的 TEI CPU 服务。
Intel/AMD CPU 使用默认 `cpu-1.9` 镜像。ARM64/Apple Silicon 在 `.env` 中改为：

```dotenv
LETHES_EMBEDDING_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-1.9
```

```bash
docker compose -f compose.yaml -f compose.dev.yaml --profile semantic up -d --build
docker compose logs -f embedding
```

首次需下载权重，缓存保存在 embedding-data 卷，不要求 GPU。就绪后从宿主机验证：

```bash
curl http://127.0.0.1:8081/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-small-zh-v1.5","input":["她想起了之前的约定。"]}'
```

在默认配置或对应模型的 `retrieval` 中填入：

```json
{
  "embedding": {
    "api_base": "http://embedding:80/v1",
    "model": "BAAI/bge-small-zh-v1.5"
  },
  "embedding_chunk_chars": 384,
  "embedding_chunk_overlap": 64,
  "embedding_query_prefix": "为这个句子生成表示以用于检索相关文章：",
  "embedding_document_prefix": "",
  "semantic_floor": 0.25
}
```

分窗覆盖完整原文再合回原始消息计分，reminder 仍引用原文。字符数不是精确 token 数；
TEI 设置 `AUTO_TRUNCATE=false` 避免悄悄丢掉后半段。超长输入报错时调小字符窗口。
辅助服务故障保留关键词召回，并在状态中标记降级；相似度阈值需按你的数据验证。

辅助服务需要 Key 时，在其配置增加 `api_key_env`，值为环境变量名，再通过你自己的
Compose environment/env_file 将对应变量传入 lethes。查询扩展配置在 `retrieval.query`，
同样接受 api_base、model、api_key_env。更多选择见 [本地模型说明](local-models.md)。

## 6. 检查聊天与观测

```bash
curl http://127.0.0.1:8000/healthz
docker compose ps
docker compose logs -f lethes
```

在 Open WebUI 发起一轮真实聊天，应出现类似：

```text
lethes · 输入 ≈12,340/27,648 token（估算） · 续接 · 召回 4
```

可信 usage 到达后更新“本次实测”和缓存读取量。召回数是当前布局引用的原始消息
总数，包括已驻留的 reminder。工具多跳整轮汇总不冒充单次请求输入量。

观测 Web 应出现对应 request_id，可查看预算和来源、版本、原文索引、reminder 位置、
未纳入数量、周期及重建原因、准备耗时、实际 usage。页面返回计数和布局，不返回聊天
正文；SQLite 为恢复布局仍保存实际消息，备份数据卷也会包含聊天内容。
没有 usage 时显示空值；“实际缓存读取”仅统计当前列表中有可信 usage 的请求。

`lethes.context` 保存提交时的不可变预算观测，`lethes.usage` 记录后续用量，
后来的校准不会覆盖原始观测。潜在公共前缀不等于真实缓存命中。

## 7. 给其他智能体的 API 契约

OpenAPI 文档：`http://127.0.0.1:8000/docs`。业务请求均携带
`Authorization: Bearer <LETHES_API_KEY>`。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/healthz` | 无 Key 的存活检查，不含聊天数据 |
| POST | `/v2/context/prepare` | 组装一次模型请求 |
| POST | `/v2/usage` | 回传单次 usage 或明确标记的整轮汇总 |
| GET | `/v2/observations?limit=100` | 最近请求，limit 1–200 |
| GET | `/v2/observations/{request_id}` | 单次请求详情 |
| GET | `/observer/` | 同源观测 Web，无外部 CDN |

prepare 最小请求示例：

```json
{
  "user_id": "stable-user-id",
  "chat_id": "stable-chat-id",
  "model": "actual-request-model-id",
  "messages": [{"role":"user","content":"完整当前历史中的消息"}],
  "reply_tokens": 4096,
  "correlation": "stable-assistant-turn-correlation"
}
```

还接受 environment 和 request_context。工作区基础模型放入
`environment.model_info.base_model_id`；工具、tool_choice、response_format，以及会在
Hook 后插入的系统消息放在 request_context，后者使用 prefix_messages 数组。
这些内容参与预算及请求身份，构造细节以附带远程 Filter 为准。

返回 request_id、messages、observation、一行 status。将返回消息发给主模型；
下一轮再次提供完整当前原始历史，包括编辑与选中的分支，不能把上次裁剪后的历史
当成原始历史。远程 Filter 已处理工具后续请求的原文恢复。

usage 请求示例：

```json
{
  "request_id": "prepare-returned-id",
  "model": "provider-returned-model-id",
  "origin": "provider_stream",
  "aggregate": false,
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 200,
    "prompt_tokens_details": {"cached_tokens": 1024}
  }
}
```

origin 支持 provider、provider_stream、outlet；单次 provider 计数优先于 outlet 回退。
同一请求的分段 usage 会合并。无 request_id 的完成回调可使用相同 correlation 关联
最近工具 hop；多次请求的 outlet 总数自动标记 aggregate。缺失 usage 不填假零，
显式估算保留 `estimated: true`。

401 检查 Key；422 检查容量、模型映射和参数；404 表示没有对应观测请求。
服务不可用时，附带 Filter 会明确停止本次准备，不静默把完整长历史发给主模型。
usage 回传失败仅记录告警，不改写模型输出。

## 8. 常规运行、升级和备份

常规运行取消源码挂载及自动重载：

```bash
docker compose up -d --build
```

需要 embedding 时用 `docker compose --profile semantic up -d --build`。
服务和 Web 是同一镜像，访问方式与数据卷不变。开发时使用第二节的双文件命令。

更新前备份，目录名每次改成新的以保留旧备份：

```bash
docker compose stop lethes
docker compose cp lethes:/data ./backup-before-update
docker compose start lethes
```

复制停止写入后的整个目录，包含可能存在的 WAL 文件。另行保管 `.env` 与本地 JSON
配置。源码有本地改动时先提交自己的分支，再合并上游；按所选模式重新 up --build。
`docker compose down` 保留命名卷；要保留聊天状态时不要追加 -v。
自定义 bind mount 需允许容器 UID 10001 写入。SQLite 用于单机，默认一个服务进程。

## 9. 常见问题

| 现象 | 检查 |
|---|---|
| 启动报 Key 未设置 | 初始化 .env，填入至少 16 字符的 Key |
| Open WebUI 连不上 | 两边加入同一网络，Filter 用 http://lethes:8000 |
| Web 列表 401 | Key 与容器不一致；修改 .env 后重新 up -d |
| Web 没请求 | 检查 service Filter、模型绑定和聊天开关；内部标题任务会跳过 |
| 没有本次实测 | 模型未返回 usage、回调未到，或返回模型不匹配；查看用量记录 |
| unknown LiteLLM model | 为实际请求 ID 设置正确 tokenizer_model |
| 422 容量不足 | 核对预留；当前保护轮次本身过大时明确拒绝 |
| 修改 Python 无反应 | 确认加载 compose.dev.yaml，日志有 reload；依赖变化需重建 |
| Web 显示旧内容 | 刷新页面，确认挂载的是当前 checkout 的 src |
| embedding 架构错误 | x86/ARM64 选择对应镜像，不强制错误平台 |
| 数据库无法写入 | 检查挂载及 UID 10001 权限 |

API、回传和重启恢复有自动化测试；Compose 可做配置校验。当前开发环境没有 Docker
daemon 操作权限，未在此完成镜像构建及容器实机启动验证。

参考（检查于 2026-11-06）：[Open WebUI v0.13.0 Hook](https://github.com/open-webui/open-webui/blob/v0.13.0/backend/open_webui/utils/middleware.py)、
[Filter 文档](https://docs.openwebui.com/features/extensibility/plugin/functions/filter/)、
[TEI 硬件与镜像](https://huggingface.co/docs/text-embeddings-inference/supported_models)、
[TEI 参数与自动截断](https://huggingface.co/docs/text-embeddings-inference/cli_arguments)。

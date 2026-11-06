# lethes v2（Alpha）

面向 Open WebUI 长聊天，尤其是小说和角色扮演的原文召回与上下文组装库。
这是一次不兼容 v1 的重写。

每轮以 Open WebUI 传入的完整当前历史为依据：保留近期连续原文，召回较早
原文片段，作为历史 reminder 补入上下文。系统不抽取世界规则，不维护角色事实，
也不反复生成滚动摘要。

- 关键词检索支持中文；可选语义检索和辅助模型查询扩展，候选取并集并补邻近原文。
- reminder 的内容和插入位置在本次缓存周期内固定，后续对话持续追加。
- 先复用已有上下文：近期原文和既有 reminder 都计入当前召回预算，已覆盖的原文不重复
  插入，也不因复用了原文就继续填入下一批较弱候选。不同事件和编辑版本保留各自来源。
- 内容、角色及祖先路径共同确定消息版本，支持编辑输入/输出、重生成、回退和分叉。
- SQLite 持久化不可变快照和独立请求记录；当前路径无法引用其他分支或未来原文。
- 配置真实价格后，按照持续聊天的重建周期平均费用判断经济重建，不预测用户退出。
- 优先采用同模型、同请求的历史真实输入 token；沿相同前缀追加时，只估算新增部分。
  计数来源、潜在公共前缀与接口实际返回的缓存用量分别记录。

## 开发与测试

```bash
uv sync --extra dev
uv run pytest
uv run python examples/replay.py --synthetic-turns 300 --output /tmp/lethes-replay.csv
uv build
```

## Open WebUI 安装

先把构建的 `dist/lethes-2.0.0a1-py3-none-any.whl` 安装到 **Open WebUI 后端环境**，
再把 `src/external/open_webui_filter.py` 导入管理后台 Functions，启用、绑定模型并在聊天中选中。
本 Alpha 尚未发布到 PyPI，需要安装本地 wheel。

Valves 中设置模型上下文容量、回复预留和近期原文预算；`state_path` 应位于持久卷。
只有配置 `input_price` 和 `cached_input_price`（美元/百万 token）后才启用经济重建。
默认使用离线 UTF-8 字节估算器。推荐安装 wheel 的 `[litellm]` 可选依赖，
设置 `tokenizer=litellm`；自动采用工作区 `base_model_id`，也可用 `tokenizer_model`
显式映射部署别名或接口返回的具体版本。未知模型不会静默按 GPT 计数。
LiteLLM 本地计数覆盖工具定义和工具选择；`response_format` 另按序列化文本预留。
SDK 计数仍是估算，不调用生成或远程计数接口。离线环境可在启动前设置
`LITELLM_LOCAL_MODEL_COST_MAP=True`，并准备所选模型需要的本地 tokenizer 资源。
也可安装 `lethes[tokenizers]` 后设置合适的 tiktoken encoding。本版重点支持文本聊天；
LiteLLM 的默认图像计数不代表精确多模态预算，字节/tiktoken 后端拒绝图像音频输入。

历史真实 usage 会持久化，重启或切换本地计数器后仍优先使用，但必须匹配模型、
工具等请求配置和实际组装消息。编辑、分叉或重建后的内容不冒充已测量请求；
只参考同配置的历史低估量作保守修正。`result.input_tokens` 是预算采用的输入量，
`count_source` 和 `count_reference` 标明来源及对应请求。真实计数更新不会触发每轮重建；
只有它使已存布局超出容量等硬约束时才修复布局。

集成依据 2026-11-06 查询的最新官方文档：

- [Filter hooks](https://docs.openwebui.com/features/extensibility/plugin/functions/filter/)
- [保留参数](https://docs.openwebui.com/features/extensibility/plugin/development/reserved-args/)

`inlet()` 保留完整历史，跳过标题等内部任务；`request()` 在流水线补充上下文后组装，
支持工具调用后续请求的幂等处理；`stream()` / `outlet()` 只记录可用的实际 usage。
优先采用单次 provider 请求计数；工具多跳的整轮汇总保留在日志中，不作为单次请求预算。
接口返回模型不匹配时停用该路由的旧校准。输出 token 不计入输入量，回复容量独立预留。
工作区模型提示词及供应商格式转换仍可能在 request 后发生，真实缓存收益需要实测。

每次请求用原生状态栏显示一行，例如：

```text
lethes · 输入 ≈12,340/27,648 token（估算） · 续接 · 召回 4
```

分母是扣除回复和安全预留后的输入上限；`≈` 表示估算或实测前缀加增量估算。
匹配的历史计数标为“历史实测”，可信的本次 usage 到达后更新为“本次实测”，
有真实缓存读取量时一并显示。未纳入的原文数量和检索降级也会在同一行标明。
召回数量统计当前布局引用的原始消息，包括已经驻留的 reminder，不代表每轮重新插入。
状态历史的折叠由 Open WebUI 控制，插件不往提示词或回复正文塞入遥测。
`show_status=false` 可关闭状态提示，不重置前缀。

后端为 `lethes.integrations.open_webui` 开启 INFO 日志，可按 request_id 查看准备耗时、
周期及重建原因、输入预算、计数来源、召回/未纳入数量、潜在公共前缀与实际 usage。
日志不包含聊天正文、查询文本、向量和密钥。SQLite 分别保存请求发出时的预算观测
`store.observation(request_id)` 与后续 `store.usage(request_id)`，重启后也能对照。

小型本地模型优先用于 embedding，现已支持查询侧指令、长消息重叠分窗和可配置相似度阈值。
分窗只影响检索表示，召回仍引用原始消息；同一消息不会因为窗多而重复计票。
推荐先评估 24M 的中文 BGE-small，再考虑 0.6B 档模型；配置和限制见
[本地模型说明](docs/local-models.md)。

## 后续数据回放

```bash
uv run python examples/replay.py --input chat.json --output /tmp/replay.csv
```

输入可为消息数组、`{"messages": [...]}`，或包含编辑/分支过程的
`{"histories": [[...], [...]]}`。脚本不调用主模型，适合先验证机制；
真实叙事效果和服务端缓存费用仍需后续聊天数据及实际环境验证。

完整参数、使用方式、限制见 [README](README.md)，机制与经济模型见 [设计说明](docs/design.md)。

## 小说原文回放

```bash
uv run python examples/novel_replay.py --text /path/to/novel.txt --output /tmp/novel-report
```

脚本逐字保留正文，以连续片段和人工“继续”消息构造机制测试负载，不调用模型生成续写。
检查容量、每段召回原文的版本、回退修改和路径恢复。报告只含哈希、数量和位置，不含小说正文。
可通过 `--probes probes.json` 提供 `[{"query":"检索问题","expected":"目标原文子串"}]`，
验证人工选定片段的召回；此类检查不能代表真实互动质量。

真实 Open WebUI 导出可直接回放：

```bash
uv run python examples/export_replay.py --input /path/to/chat-export.json --output /tmp/interactive-report
```

导入器按 `parentId` 还原每条分支，最后回到 `currentId`，不会把旁支拼成同一段聊天。
回放实际用户轮次并验证前缀、输入/输出编辑和恢复。默认使用保守字节估算，
可通过 `--tokenizer` 指定已经具备本地词表的 tiktoken encoding。
使用 LiteLLM 可传入 `--tokenizer litellm --tokenizer-model gpt-4o`；离线回放没有真实
provider usage，此时验证的是 SDK 本地计数下的机制。

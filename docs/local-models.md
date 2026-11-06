# Small local retrieval models

Start with semantic embeddings alongside lexical BM25. They can find an earlier
passage when the current action uses different wording. They do not extract story
rules or decide whether earlier dialogue is true. This is an implementation and
evaluation recommendation, not a measured model-quality improvement on our data.

## Choices to evaluate

| Component | First candidate | Role and tradeoff |
|---|---|---|
| Chinese embeddings | [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5), about 24M parameters | First CPU experiment; its [512-token encoder limit](https://huggingface.co/BAAI/bge-small-zh-v1.5/blob/main/sentence_bert_config.json) requires passage windows. |
| Longer/multilingual embeddings | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | More resources; the model card lists 32K input support and query instructions. |
| Candidate reranking | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | A later experiment; evaluating query/passage pairs costs more than looking up cached embeddings. No reranker is integrated yet. |

Query expansion with a small instruction model is already possible through the
optional OpenAI-compatible query helper, but should follow the embedding experiment.
Generated queries can misresolve references; retain the original queries and
lexical candidates. Never turn their output into story facts, a recall gate, or
an authority for pruning original text.

Keep branch identity, exact source coverage, duplicate checks, immutable reminder
placement and cache-cycle economics deterministic. A reranker, if added, should
reorder candidates while protecting direct-query anchors and preserving unscored
candidates. It must not rewrite reminders already present in a valid prefix.

## Endpoint configuration

For the standalone service, use `defaults.retrieval` or a per-model override in
`deploy/lethes.local.json`; the bundled Compose `semantic` profile starts the CPU
encoder. See the [deployment guide](deployment-cn.md). The configuration below
describes the optional embedded Filter Valves; the service nests its endpoint
under `retrieval.embedding` and retains the window/prefix options in `retrieval`.

Run the encoder as a separate local service exposing OpenAI-compatible
`POST /v1/embeddings`; no inference framework is added to Open WebUI's process.
[TEI's official guide](https://huggingface.co/docs/text-embeddings-inference/quick_tour)
documents CPU/GPU deployment and this route. Follow its supported hardware/model
instructions when deploying; no service is automatically started by lethes.

For a Chinese BGE experiment, set these Filter Valves:

```json
{
  "embedding_api_base": "http://embedding:80/v1",
  "embedding_api_key": "",
  "embedding_model": "BAAI/bge-small-zh-v1.5",
  "embedding_chunk_chars": 384,
  "embedding_chunk_overlap": 64,
  "embedding_query_prefix": "为这个句子生成表示以用于检索相关文章：",
  "embedding_document_prefix": "",
  "semantic_floor": 0.25
}
```

`embedding` above is an example service DNS name reachable from the Open WebUI
backend; choose the actual address and model ID accepted by your server. In a
container, localhost refers to that container. The endpoint determines where
text is processed; setting a local model name does not change a remote endpoint.

The Chinese prefix follows the BGE model card's short-query guidance. For Qwen,
use a query prefix such as
`Instruct: Retrieve earlier original dialogue relevant to the current scene.\nQuery: `
with an actual newline; leave the passage prefix empty. Change the instruction
for the task and evaluate it. Do not also enable an equivalent server-side prefix.

Character windows are soft input bounds, not exact model-token limits. Leave
room for query instructions/special tokens, verify the chosen tokenizer's limits,
and configure the server to reject oversized input rather than silently truncating
it. Changing the main chat tokenizer does not change the embedding tokenizer.
The BGE configuration above is a starting point, not a guarantee for every text.
Similarity distributions also differ by model: tune `semantic_floor` on positive
and negative examples; 0.25 is the existing default, not a universal calibration.

## Semantics and cost

All original characters participate in overlapping windows, including late
passages and long queries. Each original message receives at most one semantic
rank contribution per query, using its best window-pair similarity. More windows
cannot multiply that message's votes. Selected ranges still cite full original
messages with their source revisions; character-level reminder compression is
not implemented. Lexical candidates and neighboring-message expansion remain.

Embedding windows are cached by their exact prefixed text and model. The bounded
in-memory cache avoids re-encoding identical windows while resident; an edit
invalidates changed windows. Caches are not persisted across backend restarts,
so measure cold start separately. A large first history or cache eviction can
require recomputation. Query/passage instructions and nondefault window settings
isolate assembly domains; default settings preserve existing layouts.

With no endpoint configured, operation stays lexical/offline. Helper failures
retain lexical retrieval, appear in structured observations and add a short
`检索降级` marker to the one-line status. Helpers currently use a 30-second HTTP
timeout. Model weights are not bundled or downloaded by lethes.

## Evaluation before making it a default

Compare lexical-only, lexical+small embedding and then optional reranking using
the same histories, input budgets and queries. Check selected original passages,
nearby exceptions, references with changed wording, negatives and edited branches.
Include cold-start and warm p50/p95 assembly latency, cache misses/recomputation,
CPU/RAM use, omitted evidence, rebuild frequency and actual provider input/cache
usage. Do not treat potential prefixes as billed hits.

The current unit tests use controlled embedding responses to verify full window
coverage, late-passage retrieval, per-message voting, query prefixes, cache reuse
and fallback. They do not measure these pretrained models' recall or throughput.
No weights or local inference service were loaded for this change.

"""Explicit estimators; local estimates must never be labelled billed usage."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from importlib.metadata import version
from typing import Protocol

from .history import Message, canonical, digest


class TokenCounter(Protocol):
    def count(self, messages: list[Message]) -> int: ...


def request_fingerprint(context: dict) -> str:
    """Tool/schema property order may affect rendered prompts even if JSON is equivalent."""
    return digest(
        {
            key: json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            for key, value in context.items()
        }
    )


class UnsupportedContent(ValueError):
    pass


@dataclass(frozen=True)
class TokenCount:
    tokens: int
    source: str
    reference: str | None = None


class LiteLLMTokenCounter:
    """Optional local SDK counter, bound to one model and request envelope.

    SDK counts may use provider approximations. No completions or remote count
    endpoints are invoked. Image URLs are never downloaded by this adapter.
    """

    def __init__(
        self,
        model: str,
        *,
        tools: list[dict] | None = None,
        tool_choice: object = None,
        prefix_messages: list[Message] | None = None,
        response_format: dict | None = None,
    ) -> None:
        try:
            import litellm
        except ModuleNotFoundError as exc:
            if exc.name == "litellm":
                raise ImportError("Install lethes[litellm] to use LiteLLMTokenCounter") from exc
            raise RuntimeError("LiteLLM dependencies could not initialize") from exc
        if not model:
            raise ValueError("An actual tokenizer model is required")
        registry = litellm.model_cost
        sdk_model = model
        if "/" in sdk_model and (
            sdk_model not in registry or sdk_model.startswith(("openai/", "azure/"))
        ):
            candidate = sdk_model.split("/", 1)[1]
            if candidate in registry:
                sdk_model = candidate
        if sdk_model not in registry:
            raise ValueError(
                "Unknown LiteLLM model; map the deployment to a registered tokenizer_model"
            )
        self.model = model
        self.sdk_model = sdk_model
        self._sdk = litellm

        def freeze(value: object) -> str:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

        self._tools = freeze(tools)
        self._choice = freeze(tool_choice)
        self._prefix = freeze(prefix_messages or [])
        self._format = freeze(response_format)
        self.request_context = json.loads(
            freeze(
                {
                    k: v
                    for k, v in {
                        "tools": tools,
                        "tool_choice": tool_choice,
                        "prefix_messages": prefix_messages,
                        "response_format": response_format,
                    }.items()
                    if v
                }
            )
        )
        self.cache_identity = "litellm:" + digest(
            [version("litellm"), sdk_model, self._tools, self._choice, self._prefix, self._format]
        )
        self._cache: OrderedDict[str, int] = OrderedDict()

    def count(self, messages: list[Message]) -> int:
        key = canonical(messages)
        if key in self._cache:
            return self._cache[key]
        count = self._sdk.token_counter(
            model=self.sdk_model,
            messages=json.loads(self._prefix) + json.loads(key),
            tools=json.loads(self._tools),
            tool_choice=json.loads(self._choice),
            count_response_tokens=False,
            use_default_image_token_count=True,
        )
        if self._format != "null":
            # SDK token_counter has no response_format parameter: reserve its text.
            count += self._sdk.token_counter(model=self.sdk_model, text=self._format)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or (messages and count == 0)
        ):
            raise ValueError("LiteLLM returned an invalid/disabled token count")
        self._cache[key] = count
        if len(self._cache) > 512:
            self._cache.popitem(last=False)
        return count


class EnvelopeCounter:
    """Bind caller-estimated input overhead once; reply reservation is separate."""

    def __init__(self, counter: TokenCounter, overhead: int) -> None:
        self.counter, self.overhead = counter, overhead
        self.cache_identity = getattr(counter, "cache_identity", type(counter).__qualname__)

    def count(self, messages: list[Message]) -> int:
        return self.counter.count(messages) + self.overhead


@dataclass(frozen=True)
class Measurement:
    wire: str
    tokens: int
    request_id: str


class UsageAwareCounter:
    """Per-request view: exact reported usage, then measured-prefix deltas, then local.

    Changed/rebuilt layouts never inherit a lower count merely because a different
    request was cheaper. Positive observed undercounts conservatively calibrate them.
    """

    def __init__(self, counter: TokenCounter, measurements: list[Measurement], lookup) -> None:
        self.counter = counter
        self.measurements = measurements
        self.lookup = lookup
        self._cache: dict[str, TokenCount] = {}
        self._anchors: list[tuple[list[Message], Measurement, int]] | None = None

    def describe(self, messages: list[Message]) -> TokenCount:
        wire = canonical(messages)
        if wire in self._cache:
            return self._cache[wire]
        exact = self.lookup(wire) if self.measurements else None
        if exact:
            result = TokenCount(exact.tokens, "reported_usage", exact.request_id)
        else:
            local = self.counter.count(messages)
            if self._anchors is None:
                self._anchors = [
                    (json.loads(m.wire), m, self.counter.count(json.loads(m.wire)))
                    for m in self.measurements
                ]
            extensions = [
                (old, m, estimate)
                for old, m, estimate in self._anchors
                if old and len(old) < len(messages) and messages[: len(old)] == old
            ]
            if extensions:
                old, anchor, estimate = max(extensions, key=lambda a: len(a[0]))
                result = TokenCount(
                    anchor.tokens + max(0, local - estimate),
                    "reported_prefix_plus_estimate",
                    anchor.request_id,
                )
            else:
                correction, reference = max(
                    (
                        (max(0, m.tokens - estimate), m.request_id)
                        for _, m, estimate in self._anchors
                    ),
                    default=(0, None),
                )
                result = TokenCount(
                    local + correction,
                    "usage_calibrated_estimate" if correction else "local_estimate",
                    reference if correction else None,
                )
        self._cache[wire] = result
        return result

    def count(self, messages: list[Message]) -> int:
        return self.describe(messages).tokens


class ByteTokenCounter:
    """Offline conservative text estimate: UTF-8 bytes plus framing.

    Images/audio need a provider-specific counter, not URL text counting.
    """

    cache_identity = "utf8-byte-counter-v1"

    def count(self, messages: list[Message]) -> int:
        for message in messages:
            content = message.get("content")
            if message.get("images") or message.get("audio"):
                raise UnsupportedContent("Configure a provider-specific multimodal token counter")
            if isinstance(content, list) and any(
                not isinstance(p, dict) or p.get("type") != "text" for p in content
            ):
                raise UnsupportedContent("Configure a provider-specific multimodal token counter")
        return 3 + sum(8 + len(canonical(m).encode()) for m in messages)


class TiktokenCounter(ByteTokenCounter):
    """Optional text counter; framing is estimated, schemas reserved separately."""

    def __init__(self, encoding: str = "o200k_base") -> None:
        import tiktoken

        self.encoding = tiktoken.get_encoding(encoding)
        self.cache_identity = f"tiktoken:{encoding}"

    def count(self, messages: list[Message]) -> int:
        super().count(messages)
        return 3 + sum(
            8 + len(self.encoding.encode(canonical(m), disallowed_special=())) for m in messages
        )


def common_prefix_tokens(a: list[Message], b: list[Message], counter: TokenCounter) -> int:
    shared = []
    for left, right in zip(a, b, strict=False):
        if canonical(left) != canonical(right):
            break
        shared.append(left)
    return max(0, counter.count(shared) - counter.count([])) if shared else 0

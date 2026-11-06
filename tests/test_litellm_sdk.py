"""Optional real SDK contract; no generation, count endpoint, or image download."""

import importlib.util
import os

import pytest


@pytest.mark.skipif(importlib.util.find_spec("litellm") is None, reason="optional LiteLLM extra")
def test_real_litellm_local_counter_contract():
    # The SDK's bundled registry avoids fetching a remote cost map in this test.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from lethes import LiteLLMTokenCounter

    simple = LiteLLMTokenCounter("gpt-4o")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "roll",
                "description": "Roll a die",
                "parameters": {"type": "object", "properties": {"sides": {"type": "integer"}}},
            },
        }
    ]
    with_tools = LiteLLMTokenCounter("gpt-4o", tools=tools, tool_choice="auto")
    message = [{"role": "user", "content": "你好，世界。"}]
    assert 0 < simple.count(message) < 40
    assert with_tools.count(message) > simple.count(message)
    assert simple.count([]) >= 0


@pytest.mark.skipif(importlib.util.find_spec("litellm") is None, reason="optional LiteLLM extra")
def test_provider_prefixes_resolve_the_actual_openai_tokenizer():
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from lethes import LiteLLMTokenCounter

    messages = [{"role": "user", "content": "你好，世界。"}]
    native = LiteLLMTokenCounter("gpt-4o").count(messages)
    assert LiteLLMTokenCounter("openai/gpt-4o").count(messages) == native
    assert LiteLLMTokenCounter("azure/gpt-4o").count(messages) == native

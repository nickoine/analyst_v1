# Built-in
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

# Internal
from managers.dto import LLMToolCall


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str | None
    base_url: str | None
    max_output_tokens: str | int


@dataclass(frozen=True)
class LLMRequest:
    sys_prompt: str | None = None
    user_prompt: str | None = None
    input: list[dict[str, str]] | None = None
    instructions: str | None = None

    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None

    text_format: object | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None

    temperature: float | None = None
    top_p: float | None = None

    reasoning: dict[str, Any] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None


@dataclass(frozen=True)
class LLMResponse:
    """
    Normalized application-level response returned by an LLM adapter.

    Design:
    - `response_id` is needed for multi-turn continuation with Responses API.
    - `output_text` is the convenience text view of the final response.
    - `tool_calls` holds normalized function/tool calls requested by the model.
    - `parsed` stores structured parsed output when parse/structured mode is used.
    - `usage` is useful for observability and cost tracking.
    """
    response_id: str | None
    output_text: str
    tool_calls: list[LLMToolCall]
    parsed: Any
    usage: dict[str, Any] | None


@dataclass(frozen=True)
class AgentPipeline:
    analyzer_id: str
    shaper_id: str
    runner_id: str

# Built-in
from __future__ import annotations
from json import JSONDecodeError
from typing import Any

# External
from openai import OpenAI
from pydantic import ValidationError

# Internal
from app.port import LLMClient
from llm.conf import load_llm_settings
from llm.dto import LLMRequest, LLMResponse, LLMToolCall


class OllamaClient(LLMClient):
    """
    Local Ollama OpenAI-compatible Responses API adapter.

    Responsibility:
    - translate normalized LLMRequest into Ollama-compatible OpenAI payload
    - execute local SDK request through Ollama /v1 endpoint
    - normalize SDK response into application-level DTO

    Notes:
    - Ollama runs locally by default at http://localhost:11434/v1
    - API key is required by the OpenAI SDK but ignored by Ollama
    - previous_response_id is intentionally unsupported because Ollama
      supports only non-stateful Responses API execution
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    DEFAULT_API_KEY = "ollama"

    def __init__(self) -> None:
        """Initialize OpenAI SDK client against local Ollama endpoint."""
        self._llm_config = load_llm_settings()

        kwargs: dict[str, Any] = {
            "api_key": self._llm_config.api_key or self.DEFAULT_API_KEY,
            "base_url": self._llm_config.base_url or self.DEFAULT_BASE_URL,
        }

        self._client = OpenAI(**kwargs)

    @property
    def model(self) -> str | None:
        """Return configured local Ollama model name."""
        return self._llm_config.model

    def create(
        self,
        request: LLMRequest,
        *,
        previous_response_id: str | None = None,
    ) -> LLMResponse:
        """
        Execute one local Ollama Responses API request.

        Ollama's OpenAI-compatible Responses API is non-stateful.
        There `previous_response_id` is rejected at adapter boundary.
        """
        if previous_response_id is not None:
            raise ValueError(
                "OllamaClient does not support previous_response_id. "
                "Ollama Responses API compatibility is non-stateful."
            )

        payload = self._build_payload(request=request)

        raw_response = self._call_api(
            payload=payload,
            use_parse=bool(request.text_format),
        )

        return self._to_llm_response(raw_response)

    def _build_payload(
        self,
        *,
        request: LLMRequest,
    ) -> dict[str, Any]:
        """
        Build Ollama-compatible OpenAI Responses API payload from normalized request.

        Unsupported OpenAI cloud-only features are intentionally omitted instead
        of being forwarded blindly into the local runtime.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "input": request.input,
        }

        if self._llm_config.max_output_tokens is not None:
            payload["max_output_tokens"] = self._llm_config.max_output_tokens

        if request.instructions is not None:
            payload["instructions"] = request.instructions

        if request.tools is not None:
            payload["tools"] = request.tools

        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls

        if request.text_format is not None:
            payload["text_format"] = request.text_format

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        if request.top_p is not None:
            payload["top_p"] = request.top_p

        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty

        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty

        return payload

    def _call_api(
        self,
        *,
        payload: dict[str, Any],
        use_parse: bool,
    ) -> Any:
        """
        Execute local Ollama request through the OpenAI SDK.

        When structured output is configured, responses.parse() is attempted.
        This keeps the adapter contract aligned with the cloud OpenAI adapter.
        """
        if use_parse:
            return self._client.responses.parse(**payload)

        return self._client.responses.create(**payload)

    @staticmethod
    def _to_llm_response(response: Any) -> LLMResponse:
        """
        Normalize Ollama/OpenAI-compatible SDK object into application-level LLMResponse.

        For responses.parse(), parsed output is treated as the source of truth.
        output_text remains diagnostic/display text only and may be empty for
        structured outputs.

        Local Ollama usage metrics may be less complete than OpenAI cloud metrics,
        so missing usage fields are normalized safely.
        """

        def _safe_str(value: Any) -> str:
            if value is None:
                return ""

            try:
                return str(value)
            except (TypeError, ValueError, JSONDecodeError, ValidationError):
                return ""

        def _to_plain(value: Any) -> Any:
            if value is None:
                return None

            if isinstance(value, (str, int, float, bool)):
                return value

            if isinstance(value, dict):
                return {str(k): _to_plain(v) for k, v in value.items()}

            if isinstance(value, (list, tuple)):
                return [_to_plain(v) for v in value]

            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")

            if hasattr(value, "dict"):
                return value.dict()

            return str(value)

        def _extract_tool_calls(tool_output_items: list[Any]) -> list[LLMToolCall]:
            calls: list[LLMToolCall] = []

            for item in tool_output_items:
                if getattr(item, "type", None) == "function_call":
                    calls.append(
                        LLMToolCall(
                            call_id=getattr(item, "call_id", None),
                            name=getattr(item, "name", None),
                            arguments=getattr(item, "arguments", None),
                        )
                    )

            return calls

        def _extract_nested_parsed(nested_output_items: list[Any]) -> Any:
            """
            Fallback for SDK shapes where parsed content is stored under:
            response.output[*].content[*].parsed
            """
            for item in nested_output_items:
                for content in getattr(item, "content", []) or []:
                    parsed_value = getattr(content, "parsed", None)
                    if parsed_value is not None:
                        return parsed_value

            return None

        def _extract_nested_output_text(output_text_items: list[Any]) -> str:
            """
            Extract display/debug text only.
            Do not treat this as the primary structured-output source.
            """
            chunks: list[str] = []

            for item in output_text_items:
                direct_text = getattr(item, "text", None)
                if direct_text:
                    chunks.append(str(direct_text))

                for content in getattr(item, "content", []) or []:
                    text = getattr(content, "text", None)
                    if text:
                        chunks.append(str(text))

            return "\n".join(chunks).strip()

        def _get_detail_value(container: Any, key: str) -> int:
            """
            Read detail values from either SDK objects or dictionaries.
            """
            if container is None:
                return 0

            if isinstance(container, dict):
                value = container.get(key, 0)
            else:
                value = getattr(container, key, 0)

            return int(value or 0)

        def _extract_usage(raw_usage: Any) -> dict[str, Any] | None:
            if raw_usage is None:
                return None

            input_tokens = int(getattr(raw_usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(raw_usage, "output_tokens", 0) or 0)
            total_tokens = int(getattr(raw_usage, "total_tokens", 0) or 0)

            input_details = getattr(raw_usage, "input_tokens_details", None)
            output_details = getattr(raw_usage, "output_tokens_details", None)

            cached_tokens = _get_detail_value(input_details, "cached_tokens")
            reasoning_tokens = _get_detail_value(output_details, "reasoning_tokens")

            non_cached_input_tokens = max(input_tokens - cached_tokens, 0)
            visible_output_tokens_estimate = max(output_tokens - reasoning_tokens, 0)

            return {
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "non_cached_input_tokens": non_cached_input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "visible_output_tokens_estimate": visible_output_tokens_estimate,
                "total_tokens": total_tokens,
            }

        def _print_usage_metrics(resp_id: str | None, usg: dict[str, Any] | None) -> None:
            if usg is None:
                print(f"[llm_usage] provider=ollama response_id={resp_id} usage=None")
                return

            input_tokens = usg.get("input_tokens", 0)
            cached_input_tokens = usg.get("cached_input_tokens", 0)
            non_cached_input_tokens = usg.get("non_cached_input_tokens", 0)
            output_tokens = usg.get("output_tokens", 0)
            reasoning_tokens = usg.get("reasoning_tokens", 0)
            visible_output_tokens_estimate = usg.get("visible_output_tokens_estimate", 0)
            total_tokens = usg.get("total_tokens", 0)

            cache_ratio = (
                cached_input_tokens / input_tokens
                if input_tokens
                else 0.0
            )

            reasoning_ratio = (
                reasoning_tokens / output_tokens
                if output_tokens
                else 0.0
            )

            print(
                "[llm_usage] "
                "provider=ollama "
                f"response_id={resp_id} "
                f"input={input_tokens} "
                f"cached_input={cached_input_tokens} "
                f"non_cached_input={non_cached_input_tokens} "
                f"cache_ratio={cache_ratio:.2%} "
                f"output={output_tokens} "
                f"reasoning={reasoning_tokens} "
                f"visible_output_est={visible_output_tokens_estimate} "
                f"reasoning_ratio={reasoning_ratio:.2%} "
                f"total={total_tokens}"
            )

        response_id = getattr(response, "id", None)
        output_items = getattr(response, "output", []) or []

        output_parsed = getattr(response, "output_parsed", None)
        if output_parsed is None:
            output_parsed = _extract_nested_parsed(output_items)

        parsed = _to_plain(output_parsed)

        output_text = _safe_str(getattr(response, "output_text", None)).strip()
        if not output_text:
            output_text = _extract_nested_output_text(output_items)

        tool_calls = _extract_tool_calls(output_items)
        usage = _extract_usage(getattr(response, "usage", None))

        status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)

        if status == "incomplete":
            print(
                "[llm_response] "
                "provider=ollama "
                f"response_id={response_id} "
                f"status=incomplete "
                f"incomplete_details={incomplete_details}"
            )

        _print_usage_metrics(resp_id=response_id, usg=usage)

        return LLMResponse(
            response_id=response_id,
            output_text=output_text,
            tool_calls=tool_calls,
            parsed=parsed,
            usage=usage,
        )

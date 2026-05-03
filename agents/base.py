# Built-in
from __future__ import annotations
from typing import Any, Protocol, cast
import hashlib, json

# Internal
from app.port import LLMClient
from llm.dto import LLMRequest, LLMResponse
from managers.dto import AgentProfile


class PydanticModelClass(Protocol):
    @classmethod
    def model_validate(cls, obj: Any) -> Any:
        ...


class BaseAgent:
    """
    Base class for agent-side request construction.
    Responsibility:
    - keep resolved agent profile
    - build stable instructions
    - build dynamic input
    - expose optional tools
    - build normalized base request
    - call LLM client for the initial turn
    - provide simple response helpers for runtime orchestration
    """

    _supports_actions = False
    _supports_knowledge = False
    _supports_memory = False

    _parallel_tool_calls = False
    _tool_choice = "auto"
    _temperature = 0

    def __init__(self, *, llm: LLMClient, profile: AgentProfile) -> None:
        """Initialize the agent with injected LLM client and loaded profile."""
        self._llm = llm
        self._profile = profile
        self.res: LLMResponse | None = None

    @property
    def agent_id(self) -> str:
        """Return agent id."""
        return self._profile.agent_id

    @property
    def reasoning_effort(self) -> dict[str, Any]:
        return {
            "effort": self._profile.reasoner_effort,
            "summary": self._profile.reasoner_summary,
        }

    @property
    def prompt_cache_retention(self)-> str:
        return self._profile.prompt_cache

    def supports_actions(self) -> bool:
        """Return whether this agent can expose tools/actions."""
        return self._supports_actions

    def supports_memory(self) -> bool:
        """Return whether this agent uses memory block."""
        return self._supports_memory

    def supports_knowledge(self) -> bool:
        """Return whether this agent uses knowledge block."""
        return self._supports_knowledge

    def build_sys_prompt(self) -> str:
        """Build system content block for the whole run."""

        if not self._profile.system_content:
            raise ValueError("system_content is required but not provided")

        return (
            f"<SYSTEM_CONTENT>\n"
            f"{self._profile.system_content}\n"
            f"</SYSTEM_CONTENT>"
        )

        # if self.supports_knowledge() and self._profile.knowledge:
        #     parts.append(self._profile.knowledge)
        # if self.supports_memory() and self._profile.memory:
        #     parts.append(self._profile.memory)
        # if self._profile.output_contract:
        #     parts.append(f"OUTPUT CONTRACT: {self._profile.output_contract}")

    def build_tools(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Return tool definitions allowed for this agent."""
        if not self.supports_actions():
            return None
        return tools or None

    def build_request(
        self,
        *,
        ctx: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMRequest:
        """Build normalized initial request for the LLM client."""

        prompt_cache_key = self._build_prompt_cache_key(
            agent_id=self.agent_id,
            sys_prompt=self.build_sys_prompt(),
            instructions=self._profile.instructions,
            assistant=self._profile.assistant,
            tools=self.build_tools(tools=tools)
        )
        return LLMRequest(
            input=[
                {"role": "developer", "content": self.build_sys_prompt()},
                {"role": "user", "content": self.build_user_input(ctx=ctx)},
                {"role": "assistant", "content": self._profile.assistant},
            ],
            instructions=self._profile.instructions,
            tools=self.build_tools(tools=tools),
            tool_choice=self._tool_choice,
            parallel_tool_calls=self._parallel_tool_calls,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=self.prompt_cache_retention,
            text_format=self._profile.output_contract,
            # temperature=self._temperature,
            reasoning=self.reasoning_effort
        )

    @staticmethod
    def build_continuation_request(
        *,
        base_request: LLMRequest,
        tool_outputs: list[dict[str, Any]],
    ) -> LLMRequest:
        """
        Build normalized continuation request after local tool execution.

        Responsibility:
        - preserve the original stable request contract
        - replace only the dynamic input with tool outputs
        """
        return LLMRequest(
            input=tool_outputs,
            instructions=base_request.instructions,
            tools=base_request.tools,
            tool_choice=base_request.tool_choice,
            parallel_tool_calls=base_request.parallel_tool_calls,
            prompt_cache_key=getattr(base_request, "prompt_cache_key", None),
            prompt_cache_retention=base_request.prompt_cache_retention,
            text_format=base_request.text_format,
            temperature=base_request.temperature,
            top_p=base_request.top_p,
            reasoning=base_request.reasoning,
            presence_penalty=base_request.presence_penalty,
            frequency_penalty=base_request.frequency_penalty,
        )

    @staticmethod
    def _hash_stable_payload(payload: dict) -> str:
        """Compute deterministic SHA256 digest of payload."""
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def build_user_input(ctx: str) -> str:
        """Build dynamic input for the current turn."""
        return ctx

    def _build_prompt_cache_key(
        self,
        *,
        agent_id: str,
        sys_prompt: str,
        instructions: str,
        assistant: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build prompt-cache key from stable request parts."""
        stable_payload = {
            "agent_id": agent_id,
            "sys_prompt": sys_prompt,
            "instructions": instructions,
            "assistant": assistant,
            "tools": tools,
        }
        digest = self._hash_stable_payload(stable_payload)
        return f"agent:{agent_id}:{digest}"

    def validate_pydantic_model(self, data: Any) -> Any:
        """
        Validate parsed LLM output against the agent's configured Pydantic model.
        """
        if data is None:
            raise ValueError(f"{self.agent_id}: parsed output is missing")

        model_cls = self._profile.output_contract
        if model_cls is None:
            raise ValueError(f"{self.agent_id}: output_contract is not configured")

        if not hasattr(model_cls, "model_validate"):
            raise TypeError(
                f"{self.agent_id}: output_contract must be a Pydantic model class, got {type(model_cls)!r}"
            )

        typed_model_cls = cast(PydanticModelClass, model_cls)
        return typed_model_cls.model_validate(data)

    def run(
        self,
        *,
        ctx: str,
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> LLMResponse:
        """
        Execute one turn for agent.

        `previous_response_id` is optional and allows agent to continue
        within an existing Responses API conversation thread.
        """
        request = self.build_request(ctx=ctx, tools=tools)
        response = self._llm.create(
            request,
            previous_response_id=previous_response_id,
        )
        self.res = response
        return response

    @staticmethod
    def has_tool_calls(response: LLMResponse) -> bool:
        """Return whether the response contains tool calls."""
        return bool(getattr(response, "tool_calls", None))

    @staticmethod
    def extract_output_text(response: LLMResponse) -> Any:
        """Return parsed structured output when available."""
        if response.parsed is not None:
            return response.parsed
        return response.output_text

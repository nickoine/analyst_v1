# Built-in
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any


@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    profile: AgentProfile


@dataclass
class AgentProfile:
    agent_id: str
    system_content: str = ""
    user_content: str = ""
    instructions: str = ""
    assistant: str = ""
    actions: list[str] = field(default_factory=list)
    output_contract: object | None = None
    output_reference: str = ""
    knowledge: str = ""
    memory: str = ""
    evaluators: str = ""
    reasoners: str = ""
    planners: str = ""
    feedback: str = ""

    reasoner_effort: str = ""
    reasoner_summary: str = ""
    prompt_cache: str = ""

    source_path: Path | None = None


@dataclass(frozen=True)
class LLMToolCall:
    call_id: Any | None
    name: Any | None
    arguments: Any | None


@dataclass(frozen=True)
class ActionSpec:
    """
    Internal representation of one model-exposed action.

    Fields:
        name:
            Public tool name used in profile YAML and model tool calls.

        description:
            Human-readable description shown to the model.

        parameters:
            JSON Schema object describing tool arguments.

        handler:
            Python callable that executes the action.

        strict:
            Whether strict tool argument validation should be requested from the model.
    """
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    strict: bool = True

    def as_openai_tool(self) -> dict[str, Any]:
        """
        Convert ActionSpec into the flat tool definition expected by the LLM client.

        Example result:
            {
                "type": "function",
                "name": "read_input_data",
                "description": "...",
                "parameters": {...},
                "strict": True,
            }
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
        }

# Built-in
from __future__ import annotations
import json
from typing import Any

# Internal
from agents.base import BaseAgent
from llm.dto import LLMResponse
from output_contract.shaper import (
    ShaperHandoff
)

class RunnerAgent(BaseAgent):
    """Agent that converts shaped sub-task handoff into execution-ready unit-boundary test handoff."""

    _supports_actions = False
    _supports_knowledge = False
    _supports_memory = False

    _parallel_tool_calls = False
    _tool_choice = "auto"
    _temperature = 0

    @staticmethod
    def prepare_runner_input(
        handoff: ShaperHandoff,
    ) -> ShaperHandoff:
        """Prepare deterministic runner input from shaper handoff."""
        return handoff

    def build_runner_payload(
        self,
        handoff: ShaperHandoff,
    ) -> dict[str, Any]:
        """Convert prepared runner input into plain JSON-safe payload."""
        prepared = self.prepare_runner_input(handoff)
        return prepared.model_dump(mode="json")

    def build_user_input_from_handoff(
        self,
        handoff: ShaperHandoff,
    ) -> str:
        """Convert shaper handoff into serialized runner input."""
        payload = self.build_runner_payload(handoff)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def run_from_handoff(
        self,
        handoff: ShaperHandoff,
        *,
        previous_response_id: str | None = None,
    ) -> LLMResponse:
        """Run runner agent from validated shaper handoff."""
        ctx = self.build_user_input_from_handoff(handoff)
        return self.run(
            ctx=ctx,
            previous_response_id=previous_response_id,
        )

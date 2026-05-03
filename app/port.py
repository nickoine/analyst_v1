# Built-in
from __future__ import annotations
from typing import Protocol

# Internal
from app.dto import RunResult, IDECtx
from llm.dto import LLMRequest, LLMResponse


class AgentGateway(Protocol):
    """Application port for executing the full agent pipeline."""

    def run(self, *, ctx: IDECtx) -> RunResult:
        """Execute the pipeline for the supplied context."""
        ...


class LLMClient(Protocol):
    """Infrastructure port for a normalized LLM client."""

    def create(
        self,
        request: LLMRequest,
        *,
        previous_response_id: str | None = None,
    ) -> LLMResponse:
        """Send one request to the LLM, optionally chaining from a prior response."""
        ...
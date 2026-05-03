# Built-in
from __future__ import annotations

# Internal
from app.dto import IDEContextResponse, IDECtx
from app.port import AgentGateway
from app.projection import project_run_result_to_ide_response


class AnalysisService:
    """Application service for executing analysis and preparing IDE tasks."""

    def __init__(self, runtime: AgentGateway) -> None:
        """Store runtime dependency."""
        self._runtime = runtime

    def execute(self, *, ctx: IDECtx) -> IDEContextResponse:
        """
        Execute the runtime pipeline and project the raw runner output into IDE tasks.

        RuntimeAgentGateway returns the internal Runner output.
        This service owns the boundary between runtime-native data and IDE API data.
        """
        runtime_result = self._runtime.run(ctx=ctx)

        return project_run_result_to_ide_response(runtime_result)

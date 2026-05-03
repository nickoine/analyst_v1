# Built-in
from __future__ import annotations

# Internal
from app.dto import IDEContextResponse, IDECtx
from app.service import AnalysisService

from llm.client import OAIClient
from llm.dto import AgentPipeline

from managers.prompt_manager import PromptManager
from agents.gateway import RuntimeAgentGateway


class Application:
    """Application facade layer."""

    def __init__(self, analysis_service: AnalysisService) -> None:
        """Store the ready-to-use analysis service."""
        self._analysis_service = analysis_service

    def analyze_context(self, ctx: IDECtx) -> IDEContextResponse:
        """Execute the configured analysis pipeline and return IDE tasks."""
        return self._analysis_service.execute(ctx=ctx)


def build_application(
        *,
        prompts_dir: str = "prompts",
        analyzer_id: str = "analyzer_v1",
        critic_id: str = "shaper_v1",
        runner_id: str = "runner_v1",
    ) -> Application:
    """Build the application object graph."""
    prompt_manager = PromptManager(prompts_dir)
    prompt_manager.load_profiles()

    llm_client = OAIClient()

    pipeline = AgentPipeline(
        analyzer_id=analyzer_id,
        shaper_id=critic_id,
        runner_id=runner_id,
    )

    runtime = RuntimeAgentGateway(
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        pipeline=pipeline,
    )

    analysis_service = AnalysisService(runtime=runtime)

    return Application(analysis_service=analysis_service)

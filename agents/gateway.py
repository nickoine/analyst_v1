# Built-in
from __future__ import annotations
from typing import Any

# Internal
from app.port import AgentGateway
from app.dto import IDECtx, RunResult

from llm.client import OAIClient
from llm.dto import AgentPipeline, LLMResponse

from managers.prompt_manager import PromptManager
from agents import AnalyzerAgent, ShaperAgent, RunnerAgent


class RuntimeAgentGateway(AgentGateway):
    """
    Runtime adapter that orchestrates the full multi-agent pipeline.

    Pipeline:
        1. AnalyzerAgent:
            Reads input and produces AnalyzerHandoff.

        2. ShaperAgent:
            Reads AnalyzerHandoff and produces ShaperHandoff.

        3. RunnerAgent:
            Reads ShaperHandoff and produces RunnerHandoff.

    Design notes:
        - Each stage must return structured output.
        - The parsed structured response is treated as the source of truth.
        - Raw output_text is diagnostic/display-only and must not be used as the
          primary validation source for structured outputs.
        - Each handoff is validated through the owning agent before being passed
          downstream.
    """

    def __init__(
        self,
        *,
        prompt_manager: PromptManager,
        llm_client: OAIClient,
        pipeline: AgentPipeline,
        max_iter: int = 3,
    ) -> None:
        """
        Store runtime dependencies and pipeline configuration.

        Args:
            prompt_manager:
                Prompt/profile manager used to resolve agent profiles prompts by id.

            llm_client:
                LLM client adapter used by all pipeline agents.

            pipeline:
                Pipeline configuration containing analyzer, shaper, and runner ids.

            max_iter:
                Reserved iteration limit for tool loops.
                Currently, stored but not used by the linear pipeline.
        """
        self._pm = prompt_manager
        self._llm = llm_client
        self._pipeline = pipeline
        self._max_iter = max_iter

    def run(self, *, ctx: IDECtx) -> RunResult:
        """
        Execute the full Analyzer → Shaper → Runner pipeline.

        Args:
            ctx:
                IDE context wrapper containing the user/project input context.

        Returns:
            RunResult containing the final Runner output serialized as formatted JSON.

        Raises:
            ValueError:
                If any stage returns missing parsed output or invalid structured output.
        """

        analyzer = self._build_analyzer()
        analyzer_response = self._run_analyzer(
            analyzer=analyzer,
            ctx=ctx,
        )
        analyzer_handoff = analyzer.validate_pydantic_model(
            analyzer_response.parsed
        )

        shaper = self._build_shaper()
        shaper_response = self._run_shaper(
            shaper=shaper,
            analyzer_handoff=analyzer_handoff,
            previous_response_id=analyzer_response.response_id,
        )
        shaper_handoff = shaper.validate_pydantic_model(
            shaper_response.parsed
        )

        runner = self._build_runner()
        runner_response = self._run_runner(
            runner=runner,
            shaper_handoff=shaper_handoff,
            previous_response_id=shaper_response.response_id,
        )
        runner_handoff = runner.validate_pydantic_model(
            runner_response.parsed
        )

        self._print_pipeline_usage(
            analyzer_response,
            shaper_response,
            runner_response,
        )

        return RunResult(
            content=runner_handoff.model_dump(mode="json"),
        )

    def _build_analyzer(self) -> AnalyzerAgent:
        """
        Build the AnalyzerAgent from the configured analyzer profile.

        Returns:
            Initialized AnalyzerAgent.
        """
        analyzer = AnalyzerAgent(
            llm=self._llm,
            profile=self._pm.require(self._pipeline.analyzer_id),
        )
        print(f"[runtime] analyzer agent_id={analyzer.agent_id}")
        return analyzer

    def _build_shaper(self) -> ShaperAgent:
        """
        Build the ShaperAgent from the configured shaper profile.

        Returns:
            Initialized ShaperAgent.
        """
        shaper = ShaperAgent(
            llm=self._llm,
            profile=self._pm.require(self._pipeline.shaper_id),
        )
        print(f"[runtime] shaper agent_id={shaper.agent_id}")
        return shaper

    def _build_runner(self) -> RunnerAgent:
        """
        Build the RunnerAgent from the configured runner profile.

        Returns:
            Initialized RunnerAgent.
        """
        runner = RunnerAgent(
            llm=self._llm,
            profile=self._pm.require(self._pipeline.runner_id),
        )
        print(f"[runtime] runner agent_id={runner.agent_id}")
        return runner

    def _run_analyzer(
        self,
        *,
        analyzer: AnalyzerAgent,
        ctx: IDECtx,
    ) -> LLMResponse:
        """
        Run AnalyzerAgent and validate that a structured parsed response exists.

        Args:
            analyzer:
                Analyzer agent instance.

            ctx:
                Full context for analyzer input.

        Returns:
            LLMResponse from the analyzer stage.

        Raises:
            ValueError:
                If analyzer parsed output is missing.
        """
        print("[runtime] starting ANALYZER stage")

        response = analyzer.run(
            ctx=ctx.input_ctx,
        )

        print(f"[runtime] analyzer response_id={response.response_id}")
        self._print_stage_usage("analyzer", response)
        self._assert_parsed("Analyzer", response)

        return response

    def _run_shaper(
        self,
        *,
        shaper: ShaperAgent,
        analyzer_handoff: Any,
        previous_response_id: str | None,
    ) -> LLMResponse:
        """
        Run ShaperAgent from AnalyzerHandoff.

        Args:
            shaper:
                Shaper agent instance.

            analyzer_handoff:
                Validated AnalyzerHandoff produced by the analyzer stage.

            previous_response_id:
                Response id of the analyzer stage. This may be passed to the LLM
                provider for response chaining when enabled by the agent/client.

        Returns:
            LLMResponse from the shaper stage.

        Raises:
            ValueError:
                If shaper parsed output is missing.
        """
        print("[runtime] starting SHAPER stage")

        response = shaper.run_from_handoff(
            handoff=analyzer_handoff,
            previous_response_id=previous_response_id,
        )

        print(f"[runtime] shaper response_id={response.response_id}")
        self._print_stage_usage("shaper", response)
        self._assert_parsed("Shaper", response)

        return response

    def _run_runner(
        self,
        *,
        runner: RunnerAgent,
        shaper_handoff: Any,
        previous_response_id: str | None,
    ) -> LLMResponse:
        """
        Run RunnerAgent from ShaperHandoff.

        Args:
            runner:
                Runner agent instance.

            shaper_handoff:
                Validated ShaperHandoff produced by the shaper stage.

            previous_response_id:
                Response id of the shaper stage. This may be passed to the LLM
                provider for response chaining when enabled by the agent/client.

        Returns:
            LLMResponse from the runner stage.

        Raises:
            ValueError:
                If runner parsed output is missing.
        """
        print("[runtime] starting RUNNER stage")

        response = runner.run_from_handoff(
            handoff=shaper_handoff,
            previous_response_id=previous_response_id,
        )

        print(f"[runtime] runner response_id={response.response_id}")
        self._print_stage_usage("runner", response)
        self._assert_parsed("Runner", response)

        return response

    @staticmethod
    def _assert_parsed(stage: str, response: LLMResponse) -> None:
        """
        Fail fast when a stage did not return parsed structured output.

        Args:
            stage:
                Human-readable pipeline stage name.

            response:
                Normalized LLM response.

        Raises:
            ValueError:
                If response.parsed is None.
        """
        if response.parsed is not None:
            return

        diagnostic_text = response.output_text or ""

        print(f"[runtime][{stage}] invalid structured output")
        print(f"[runtime][{stage}] response_id={response.response_id}")
        print(f"[runtime][{stage}] usage={response.usage}")
        print(f"[runtime][{stage}] output_text_len={len(diagnostic_text)}")

        if diagnostic_text:
            print(f"[runtime][{stage}] output_text_head={diagnostic_text[:500]!r}")
            print(f"[runtime][{stage}] output_text_tail={diagnostic_text[-500:]!r}")

        raise ValueError(f"{stage} returned invalid structured output: parsed is None")

    @staticmethod
    def _print_stage_usage(stage: str, response: LLMResponse) -> None:
        """
        Print compact token usage for one pipeline stage.

        Args:
            stage:
                Stage label used in logs.

            response:
                Normalized LLM response containing usage information.
        """
        usage = response.usage or {}

        print(
            f"[runtime][{stage}] usage "
            f"input={usage.get('input_tokens')} "
            f"cached_input={usage.get('cached_input_tokens')} "
            f"non_cached_input={usage.get('non_cached_input_tokens')} "
            f"output={usage.get('output_tokens')} "
            f"reasoning={usage.get('reasoning_tokens')} "
            f"visible_output_est={usage.get('visible_output_tokens_estimate')} "
            f"total={usage.get('total_tokens')}"
        )

    @staticmethod
    def _sum_usage(*responses: LLMResponse) -> dict[str, int]:
        """
        Sum token usage across multiple LLM responses.

        Args:
            *responses:
                LLM responses from pipeline stages.

        Returns:
            Dictionary with aggregated token usage counters.
        """
        keys = [
            "input_tokens",
            "cached_input_tokens",
            "non_cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "visible_output_tokens_estimate",
            "total_tokens",
        ]

        totals = {key: 0 for key in keys}

        for response in responses:
            usage = response.usage or {}
            for key in keys:
                totals[key] += int(usage.get(key, 0) or 0)

        return totals

    @classmethod
    def _print_pipeline_usage(cls, *responses: LLMResponse) -> None:
        """
        Print total token usage for the full pipeline run.

        Args:
            *responses:
                LLM responses from all executed stages.
        """
        usage = cls._sum_usage(*responses)

        print(
            "[runtime][pipeline] usage "
            f"input={usage['input_tokens']} "
            f"cached_input={usage['cached_input_tokens']} "
            f"non_cached_input={usage['non_cached_input_tokens']} "
            f"output={usage['output_tokens']} "
            f"reasoning={usage['reasoning_tokens']} "
            f"visible_output_est={usage['visible_output_tokens_estimate']} "
            f"total={usage['total_tokens']}"
        )

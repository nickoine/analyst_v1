# Built-in
from __future__ import annotations
from typing import Any, Iterable
import json

# Internal
from agents.base import BaseAgent
from llm.dto import LLMResponse
from output_contract.analyzer import (
    AnalyzerHandoff,
    FitLevel,
    Ref,
    Trajectory,
)
from output_contract.shaper import (
    AnchorSource,
    NormalizedPlanningFrame,
    PlanningAnchor,
    PlanningEntryPoint,
    PlanningTask,
    SelectedTrajectory,
    VerificationItem,
    VerificationKind,
)


class ShaperAgent(BaseAgent):
    """Agent that converts(shapes) analyzer handoff into deterministic planning input."""

    _supports_actions = False
    _supports_knowledge = True
    _supports_memory = False

    _parallel_tool_calls = False
    _tool_choice = "auto"
    _temperature = 0

    def prepare_planning_frame(
        self,
        handoff: AnalyzerHandoff,
    ) -> NormalizedPlanningFrame:
        """Build deterministic planning frame from analyzer handoff."""
        selected = self._get_selected_trajectory(handoff)

        task = PlanningTask(
            raw=handoff.task.raw,
            goal=handoff.task.goal,
            scope=handoff.task.scope,
        )

        selected_trajectory = SelectedTrajectory(
            id=selected.id,
            name=selected.name,
            summary=selected.summary,
            rationale=selected.rationale,
            policy=handoff.decision.policy,
            components=selected.components,
            constraints=selected.constraints,
        )

        entry_point = PlanningEntryPoint(
            ref=handoff.decision.entry_point.ref,
            change_kind=handoff.decision.entry_point.change_kind,
            rationale=handoff.decision.entry_point.rationale,
        )

        anchors: list[PlanningAnchor] = []

        for item in handoff.context:
            if item.relevance in {FitLevel.high, FitLevel.medium}:
                anchors.append(
                    PlanningAnchor(
                        source=AnchorSource.context,
                        ref=item.ref,
                        role=item.role,
                        relevance=item.relevance,
                        notes=None,
                    )
                )

        for ext in handoff.extensions:
            if ext.fit in {FitLevel.high, FitLevel.medium}:
                anchors.append(
                    PlanningAnchor(
                        source=AnchorSource.extension,
                        ref=ext.ref,
                        role="extension surface",
                        relevance=ext.fit,
                        notes=ext.why_valid,
                    )
                )

        anchors.append(
            PlanningAnchor(
                source=AnchorSource.entry_point,
                ref=handoff.decision.entry_point.ref,
                role="selected entry point",
                relevance=FitLevel.high,
                notes=handoff.decision.entry_point.change_kind,
            )
        )

        for ref in selected.refs:
            anchors.append(
                PlanningAnchor(
                    source=AnchorSource.trajectory_ref,
                    ref=ref,
                    role="selected trajectory evidence",
                    relevance=selected.fit,
                    notes=None,
                )
            )

        verification_items: list[VerificationItem] = []

        for assumption in handoff.task.assumptions:
            verification_items.append(
                VerificationItem(
                    kind=VerificationKind.assumption,
                    statement=assumption,
                )
            )

        for precondition in handoff.decision.entry_point.preconditions:
            verification_items.append(
                VerificationItem(
                    kind=VerificationKind.precondition,
                    statement=precondition,
                )
            )

        return NormalizedPlanningFrame(
            task=task,
            selected_trajectory=selected_trajectory,
            entry_point=entry_point,
            anchors=self._dedupe_anchors(anchors),
            verification_items=verification_items,
            constraints=handoff.constraints,
            affected_components=self._dedupe_strings(selected.components),
            confidence=handoff.confidence,
            confidence_label=handoff.confidence_label,
        )

    def build_shaper_payload(
        self,
        handoff: AnalyzerHandoff,
    ) -> dict[str, Any]:
        """Convert planning frame into plain JSON-safe payload."""
        frame = self.prepare_planning_frame(handoff)
        return frame.model_dump(mode="json")

    def build_user_input_from_handoff(
        self,
        handoff: AnalyzerHandoff,
    ) -> str:
        """Convert analyzer handoff into serialized shaper input."""
        payload = self.build_shaper_payload(handoff)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def run_from_handoff(
        self,
        handoff: AnalyzerHandoff,
        *,
        previous_response_id: str | None = None,
    ) -> LLMResponse:
        """Run shaper agent from validated analyzer handoff."""
        ctx = self.build_user_input_from_handoff(handoff)
        return self.run(
            ctx=ctx,
            previous_response_id=previous_response_id,
        )

    @staticmethod
    def _get_selected_trajectory(handoff: AnalyzerHandoff) -> Trajectory:
        """Return selected trajectory from analyzer handoff."""
        for trajectory in handoff.trajectories:
            if trajectory.id == handoff.decision.trajectory_id:
                return trajectory

        raise ValueError(
            f"Selected trajectory_id '{handoff.decision.trajectory_id}' "
            "not found in AnalyzerHandoff.trajectories"
        )

    @staticmethod
    def _dedupe_strings(values: Iterable[str]) -> list[str]:
        """Return unique strings preserving original order."""
        seen: set[str] = set()
        result: list[str] = []

        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)

        return result

    @staticmethod
    def _ref_identity(ref: Ref) -> tuple[Any, ...]:
        """Build stable identity tuple for ref deduplication."""
        return (
            ref.file,
            ref.symbol,
            ref.kind,
            ref.line_start,
            ref.line_end,
        )

    def _dedupe_anchors(
        self,
        values: Iterable[PlanningAnchor],
    ) -> list[PlanningAnchor]:
        """Return unique anchors preserving original order."""
        seen: set[tuple[Any, ...]] = set()
        result: list[PlanningAnchor] = []

        for item in values:
            key = (
                item.source,
                item.role,
                item.relevance,
                self._ref_identity(item.ref),
            )
            if key not in seen:
                seen.add(key)
                result.append(item)

        return result

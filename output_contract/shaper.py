# Built-in
from __future__ import annotations
from typing import List, Optional
from enum import Enum

# External
from pydantic import Field, confloat

# Internal
from output_contract.analyzer import (
    StrictBaseModel,
    ExtensionPolicy,
    Ref,
    FitLevel,
    ConstraintItem,
    ConfidenceLevel
)


class PlanningTask(StrictBaseModel):
    raw: str = Field(..., description="Original task preserved exactly.")
    goal: str = Field(..., description="Planning-ready goal preserved from analyzer.")
    scope: List[str] = Field(default_factory=list, description="Scope boundaries for engineering planning.")

class SelectedTrajectory(StrictBaseModel):
    id: str = Field(..., description="Selected trajectory id.")
    name: str = Field(..., description="Selected trajectory name.")
    summary: str = Field(..., description="Selected implementation path summary.")
    rationale: str = Field(..., description="Why this path was selected.")
    policy: ExtensionPolicy = Field(..., description="Reuse/new-structure policy inherited from analyzer.")
    components: List[str] = Field(default_factory=list, description="Affected files/components.")
    constraints: List[str] = Field(default_factory=list, description="Path-specific planning constraints.")

class PlanningEntryPoint(StrictBaseModel):
    ref: Ref
    change_kind: str = Field(..., description="Nature of the first implementation change.")
    rationale: str = Field(..., description="Why implementation should begin here.")

class VerificationKind(str, Enum):
    assumption = "assumption"
    precondition = "precondition"

class AnchorSource(str, Enum):
    context = "context"
    extension = "extension"
    entry_point = "entry_point"
    trajectory_ref = "trajectory_ref"

class PlanningAnchor(StrictBaseModel):
    source: AnchorSource = Field(..., description="Origin of this anchor in the analyzer handoff.")
    ref: Ref
    role: str = Field(..., description="Why this anchor matters to the engineering plan.")
    relevance: FitLevel = Field(..., description="Anchor importance for planning.")
    notes: Optional[str] = Field(default=None, description="Optional normalized implementation note.")


class VerificationItem(StrictBaseModel):
    kind: VerificationKind = Field(..., description="Why this check exists.")
    statement: str = Field(..., description="Concrete item to verify before or during implementation.")


class NormalizedPlanningFrame(StrictBaseModel):
    """
    Minimal backend-normalized planning input for the task shaper.

    Keeps only the AnalyzerHandoff data that materially affects
    engineering plan generation.
    """

    task: PlanningTask
    selected_trajectory: SelectedTrajectory
    entry_point: PlanningEntryPoint
    anchors: List[PlanningAnchor] = Field(default_factory=list)
    verification_items: List[VerificationItem] = Field(default_factory=list)
    constraints: List[ConstraintItem] = Field(default_factory=list)
    affected_components: List[str] = Field(default_factory=list)
    confidence: confloat(ge=0.0, le=1.0) #type: ignore
    confidence_label: ConfidenceLevel


class ShapedSubTask(StrictBaseModel):
    """
    One atomic, requirement-driven engineering sub-task derived from the internal engineering plan.
    """
    id: str = Field(..., description="Stable sub-task identifier, e.g. T1, T2.")
    title: str = Field(..., description="Short atomic requirement title.")
    flow_role: str = Field(
        ...,
        description="How this sub-task advances the workflow toward the intended completion state.",
    )
    precondition: str = Field(
        ...,
        description="What must already be true before this sub-task starts.",
    )
    requirement: str = Field(
        ...,
        description="Exactly one atomic engineering obligation to execute now.",
    )
    expected_result: str = Field(
        ...,
        description="The new implementation state established by completion.",
    )
    verification: str = Field(
        ...,
        description="How completion is checked by inspection, test, or other concrete evidence.",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Ordered list of predecessor sub-task ids required before execution.",
    )
    traceability: List[str] = Field(
        default_factory=list,
        description="Frame elements that justify why this sub-task exists.",
    )


class ShaperHandoff(StrictBaseModel):
    """
    Structured output handoff produced by the task shaper.
    """
    sub_tasks: List[ShapedSubTask] = Field(
        ...,
        description="Ordered decomposition of atomic, requirement-driven engineering sub-tasks.",
        min_length=1,
    )
    confidence: confloat(ge=0.0, le=1.0) = Field( #type: ignore
        ...,
        description="Overall shaper confidence in the handoff.",
    )
    confidence_label: ConfidenceLevel = Field(
        ...,
        description="Human-readable confidence bucket.",
    )

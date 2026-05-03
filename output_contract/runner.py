# Built-in
from __future__ import annotations
from enum import Enum
from typing import List

# Internal
from output_contract.analyzer import (
    StrictBaseModel,
    ConfidenceLevel
)

# External
from pydantic import Field, confloat


class BoundaryType(str, Enum):
    EXTERNAL = "external_boundary_unit"
    INTERNAL = "internal_boundary_unit"


class ShapedSubTask(StrictBaseModel):
    """
    One atomic, requirement-driven engineering shaped sub-task produced by Shaper.
    Preserved intact inside Runner output for traceability.
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


class TestContour(StrictBaseModel):
    """
    Minimal unit-level test contract that defines what must be verified
    by the concrete unit test for this sub-task.
    """
    test_target_boundary: str = Field(
        ...,
        description="Exact unit-level surface to be tested directly.",
    )
    behavioral_expectation: str = Field(
        ...,
        description="Mandatory behavioral invariant that the test must assert.",
    )
    required_test_doubles: List[str] = Field(
        default_factory=list,
        description="Mocks, stubs, fakes, spies, fixtures, or seams required for isolation.",
    )
    success_condition: str = Field(
        ...,
        description="Precise unit-level condition that downstream implementation must satisfy.",
    )
    excluded_scope: List[str] = Field(
        default_factory=list,
        description="Concerns intentionally kept outside this unit boundary.",
    )


class RunnerSubTaskUnit(StrictBaseModel):
    """
    Execution-ready unit-boundary handoff produced by Runner for exactly one sub-task.
    """
    shaped_sub_task: ShapedSubTask = Field(
        ...,
        description="Original engineering sub-task preserved for traceability.",
    )
    boundary_framing_mode: BoundaryType = Field(
        ...,
        description="How the sub-task is framed at the unit boundary.",
    )
    implementation_boundary: str = Field(
        ...,
        description="Minimal implementation surface that downstream production code must realize.",
    )
    implementation_start_point: str = Field(
        ...,
        description="Where downstream coding should begin inside the selected unit boundary.",
    )
    test_contour: TestContour = Field(
        ...,
        description="Minimal unit-level test contour for this sub-task.",
    )
    unit_test: str = Field(
        ...,
        description="One concrete unit test encoding the mandatory behavioral invariant of the sub-task.",
    )


class RunnerHandoff(StrictBaseModel):
    """
    Structured output handoff produced by Runner.

    Contains the ordered list of shaped sub-tasks enriched with execution-ready
    unit-boundary testing artifacts.
    """
    sub_task_units: List[RunnerSubTaskUnit] = Field(
        ...,
        description="Ordered list of shaped sub-task and unit-boundary handoff pairs.",
        min_length=1,
    )
    confidence: confloat(ge=0.0, le=1.0) = Field(  # type: ignore
        ...,
        description="Overall runner confidence in the handoff.",
    )
    confidence_label: ConfidenceLevel = Field(
        ...,
        description="Human-readable confidence bucket.",
    )

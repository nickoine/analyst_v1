# Built-in
from __future__ import annotations
from typing import List, Optional
from enum import Enum

# External
from pydantic import BaseModel, Field, ConfigDict, conlist, confloat


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class SymbolKind(str, Enum):
    module = "module"
    class_ = "class"
    function = "function"
    method = "method"
    handler = "handler"
    service = "service"
    model = "model"
    schema = "schema"
    repository = "repository"
    adapter = "adapter"
    interface = "interface"
    route = "route"
    config = "config"
    task = "task"
    worker = "worker"


class FitLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class ExtensionPolicy(str, Enum):
    existing_only = "existing_only"
    existing_preferred_new_allowed = "existing_preferred_new_allowed"
    new_structure_justified = "new_structure_justified"


class Ref(StrictBaseModel):
    """
    Atomic evidence pointer into the blueprint.
    """

    file: str = Field(..., description="Repository-relative file path.")
    symbol: Optional[str] = Field(default=None, description="Referenced symbol name if applicable.")
    kind: Optional[SymbolKind] = Field(default=None, description="Kind of referenced symbol if known.")
    line_start: Optional[int] = Field(default=None, ge=1, description="Start line if known.")
    line_end: Optional[int] = Field(default=None, ge=1, description="End line if known.")
    why: str = Field(..., description="Why this artifact matters for architectural placement.")


class Task(StrictBaseModel):
    """
    Normalized architectural understanding of the incoming task.
    """

    raw: str = Field(..., description="Original technical task exactly as received.")
    goal: str = Field(..., description="Short architectural restatement of what must be placed.")
    scope: List[str] = Field(default_factory=list, description="Architectural scope items.")
    assumptions: List[str] = Field(default_factory=list, description="Grounded assumptions.")


class ContextItem(StrictBaseModel):
    """
    Filtered architectural context that materially affects task placement.
    """

    ref: Ref
    role: str = Field(..., description="Architectural role of the referenced artifact.")
    relevance: FitLevel = Field(..., description="How strongly this item affects placement.")


class Extension(StrictBaseModel):
    """
    Valid current extension point for integrating the task.
    """

    ref: Ref
    why_valid: str = Field(..., description="Why this is a legitimate extension point.")
    constraints: List[str] = Field(default_factory=list, description="Local rules at this extension point.")
    fit: FitLevel = Field(..., description="How well this extension point fits the task.")


class Trajectory(StrictBaseModel):
    """
    One coherent architectural path for integrating the task.
    """

    id: str = Field(..., description="Stable trajectory identifier.")
    name: str = Field(..., description="Human-readable trajectory name.")
    summary: str = Field(..., description="Concise explanation of this architectural path.")
    fit: FitLevel = Field(..., description="How well this trajectory fits the project.")
    rationale: str = Field(..., description="Why this path is viable according to the blueprint.")
    components: List[str] = Field(default_factory=list, description="Likely affected files or components.")
    constraints: List[str] = Field(default_factory=list, description="Architectural rules for this path.")
    advantages: List[str] = Field(default_factory=list, description="Benefits of this path.")
    drawbacks: List[str] = Field(default_factory=list, description="Tradeoffs or reasons not to choose it.")
    refs: List[Ref] = Field(default_factory=list, description="Evidence supporting this trajectory.")


class EntryPoint(StrictBaseModel):
    """
    Best initial architectural anchor for downstream work.
    """

    ref: Ref
    change_kind: str = Field(..., description="Structural nature of the first change.")
    rationale: str = Field(..., description="Why this is the strongest entry point.")
    preconditions: List[str] = Field(default_factory=list, description="Checks to confirm before expansion.")


class Decision(StrictBaseModel):
    """
    Final architectural judgment produced by the analyzer.
    """

    trajectory_id: str = Field(..., description="Identifier of the selected trajectory.")
    entry_point: EntryPoint
    policy: ExtensionPolicy = Field(..., description="Reuse/new-structure policy judgment.")
    summary: str = Field(..., description="Compact final statement of the architectural judgment.")
    rejected: List[str] = Field(default_factory=list, description="Why alternatives were not selected.")


class ConstraintItem(StrictBaseModel):
    """
    Hard constraint that downstream work must respect.
    """

    rule: str = Field(..., description="Constraint to preserve.")
    reason: str = Field(..., description="Why this constraint exists.")
    refs: List[Ref] = Field(default_factory=list, description="Evidence supporting the constraint.")


class AnalyzerHandoff(StrictBaseModel):
    """
    Structured handoff from analyzer to downstream task-expansion agent.

    """

    task: Task = Field(..., description="Normalized architectural understanding of the task.")

    context: List[ContextItem] = Field(
        default_factory=list,
        description="Filtered architectural context relevant to placement.",
    )

    extensions: List[Extension] = Field(
        default_factory=list,
        description="Valid architectural extension points grounded in the blueprint.",
    )

    trajectories: conlist(Trajectory, min_length=1) = Field( #type: ignore
        ...,
        description="Feasible architectural integration paths for the task.",
    )

    decision: Decision = Field(
        ...,
        description="Analyzer's final architectural judgment.",
    )

    constraints: List[ConstraintItem] = Field(
        default_factory=list,
        description="Hard constraints the downstream agent must respect.",
    )

    confidence: confloat(ge=0.0, le=1.0) = Field( #type: ignore
        ...,
        description="Overall analyzer confidence in the handoff.",
    )

    confidence_label: ConfidenceLevel = Field(
        ...,
        description="Human-readable confidence bucket.",
    )

# Built-in
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IDECtx:
    input_ctx: str
    task: dict[str, Any]
    blueprint: str
    overview: str


@dataclass(frozen=True)
class RunResult:
    """
    Internal runtime result.
    It carries the validated RunnerHandoff serialized into a JSON-compatible dict.
    """
    content: dict[str, Any]


@dataclass(frozen=True)
class IDEImplementationContext:
    """Implementation placement and boundary for one task."""
    boundary: str
    start_point: str


@dataclass(frozen=True)
class IDETaskInstruction:
    """Readable task instruction."""
    precondition: str
    requirement: str
    definition_of_done: str
    verification: str


@dataclass(frozen=True)
class IDETestContract:
    """Testing boundary and expected behavior for one task."""
    target_boundary: str
    behavioral_expectation: str
    success_condition: str
    required_test_doubles: list[str]
    excluded_scope: list[str]


@dataclass(frozen=True)
class IDEExecutionPolicy:
    """
    Execution policy for the agent.

    This makes it explicit that unit_test is not the final deliverable.
    The agent must write/insert the test first, then implement production code.
    """
    mode: str = "test_first_implementation"
    steps: list[str] = field(default_factory=lambda: [
        "Add or update the unit test from unit_test.",
        "Use the test as the executable contract for the task.",
        "Implement the production code required to make the test pass.",
        "Respect implementation.boundary and test_contract.excluded_scope."
    ])
    completion_rule: str = (
        "The task is complete only when the production implementation satisfies "
        "task.definition_of_done and the unit test passes."
    )


@dataclass(frozen=True)
class IDETaskUnit:
    """Single execution-ready task for the coding agent."""
    id: str
    title: str
    depends_on: list[str]
    boundary_mode: str
    implementation: IDEImplementationContext
    task: IDETaskInstruction
    test_contract: IDETestContract
    execution_policy: IDEExecutionPolicy
    unit_test: str


@dataclass(frozen=True)
class IDEContextResponse:
    """Public response contract returned to the plugin."""
    tasks: list[IDETaskUnit]

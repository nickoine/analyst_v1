# Built-in
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Internal
from app.dto import (
    IDEContextResponse,
    IDEExecutionPolicy,
    IDEImplementationContext,
    IDETaskInstruction,
    IDETaskUnit,
    IDETestContract,
    RunResult,
)


def project_run_result_to_ide_response(result: RunResult) -> IDEContextResponse:
    """
    Project internal Runtime/Runner output into the public execution contract.

    This function intentionally does not expose the raw RunnerHandoff shape.
    It maps agent-native fields into native task units.

    Args:
        result:
            Internal RunResult produced by RuntimeAgentGateway.

    Returns:
        IDEContextResponse containing execution-ready tasks.

    Raises:
        TypeError:
            If the runtime result has an invalid structural type.

        ValueError:
            If required Runner output fields are missing or empty.
    """
    content = _require_mapping(result.content, "RunResult.content")

    raw_units = content.get("sub_task_units")
    if not isinstance(raw_units, list):
        raise TypeError("RunResult.content.sub_task_units must be a list")

    if not raw_units:
        raise ValueError("RunResult.content.sub_task_units must not be empty")

    tasks: list[IDETaskUnit] = []

    for index, raw_unit in enumerate(raw_units):
        path = f"RunResult.content.sub_task_units[{index}]"
        unit = _require_mapping(raw_unit, path)

        shaped = _require_mapping_field(
            unit,
            "shaped_sub_task",
            path,
        )

        test_contour = _require_mapping_field(
            unit,
            "test_contour",
            path,
        )

        tasks.append(
            IDETaskUnit(
                id=_require_str_field(shaped, "id", f"{path}.shaped_sub_task"),
                title=_require_str_field(shaped, "title", f"{path}.shaped_sub_task"),
                depends_on=_require_str_list_field(
                    shaped,
                    "depends_on",
                    f"{path}.shaped_sub_task",
                ),
                boundary_mode=_require_str_field(
                    unit,
                    "boundary_framing_mode",
                    path,
                ),
                implementation=IDEImplementationContext(
                    boundary=_require_str_field(
                        unit,
                        "implementation_boundary",
                        path,
                    ),
                    start_point=_require_str_field(
                        unit,
                        "implementation_start_point",
                        path,
                    ),
                ),
                task=IDETaskInstruction(
                    precondition=_require_str_field(
                        shaped,
                        "precondition",
                        f"{path}.shaped_sub_task",
                    ),
                    requirement=_require_str_field(
                        shaped,
                        "requirement",
                        f"{path}.shaped_sub_task",
                    ),
                    definition_of_done=_require_str_field(
                        shaped,
                        "expected_result",
                        f"{path}.shaped_sub_task",
                    ),
                    verification=_require_str_field(
                        shaped,
                        "verification",
                        f"{path}.shaped_sub_task",
                    ),
                ),
                test_contract=IDETestContract(
                    target_boundary=_require_str_field(
                        test_contour,
                        "test_target_boundary",
                        f"{path}.test_contour",
                    ),
                    behavioral_expectation=_require_str_field(
                        test_contour,
                        "behavioral_expectation",
                        f"{path}.test_contour",
                    ),
                    success_condition=_require_str_field(
                        test_contour,
                        "success_condition",
                        f"{path}.test_contour",
                    ),
                    required_test_doubles=_require_str_list_field(
                        test_contour,
                        "required_test_doubles",
                        f"{path}.test_contour",
                    ),
                    excluded_scope=_require_str_list_field(
                        test_contour,
                        "excluded_scope",
                        f"{path}.test_contour",
                    ),
                ),
                execution_policy=IDEExecutionPolicy(),
                unit_test=_require_str_field(unit, "unit_test", path),
            )
        )

    return IDEContextResponse(tasks=tasks)


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Require a JSON object-like mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")

    return value


def _require_mapping_field(
    source: Mapping[str, Any],
    field_name: str,
    path: str,
) -> Mapping[str, Any]:
    """Require a nested object field."""
    value = source.get(field_name)

    if not isinstance(value, Mapping):
        raise TypeError(f"{path}.{field_name} must be an object")

    return value


def _require_str_field(
    source: Mapping[str, Any],
    field_name: str,
    path: str,
) -> str:
    """Require a non-empty string field."""
    value = source.get(field_name)

    if not isinstance(value, str):
        raise TypeError(f"{path}.{field_name} must be a string")

    if not value.strip():
        raise ValueError(f"{path}.{field_name} must not be empty")

    return value


def _require_str_list_field(
    source: Mapping[str, Any],
    field_name: str,
    path: str,
) -> list[str]:
    """Require a list of strings."""
    value = source.get(field_name)

    if not isinstance(value, list):
        raise TypeError(f"{path}.{field_name} must be a list")

    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f"{path}.{field_name}[{index}] must be a string")

    return list(value)

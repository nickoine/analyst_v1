# Built-in
from __future__ import annotations
from typing import Any
import json

# External
from werkzeug.datastructures import FileStorage

# Internal
from app.dto import IDECtx


import json
from typing import Any


def normalize_context(task: Any, blueprint: Any, overview: Any) -> IDECtx:
    _task = _normalize_task(task)
    _blueprint = _normalize_text(blueprint)
    _overview = _normalize_text(overview)
    _input_ctx = _merge_context(_task, _blueprint, _overview)

    return IDECtx(
        input_ctx=_input_ctx,
        task=_task,
        blueprint=_blueprint,
        overview=_overview,
    )


def _normalize_task(task: Any) -> dict[str, Any]:
    """Normalize task payload into a dictionary."""
    if isinstance(task, dict):
        return task

    if isinstance(task, str):
        return json.loads(task)

    if isinstance(task, bytes):
        return json.loads(task.decode("utf-8", errors="replace"))

    if hasattr(task, "read"):
        raw = task.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)

    if hasattr(task, "stream"):
        raw = task.stream.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)

    raise TypeError(f"Unsupported task type: {type(task)!r}")


def _normalize_text(value: Any) -> str:
    """Normalize uploaded text payload into UTF-8 string."""
    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if hasattr(value, "read"):
        raw = value.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return raw

    if hasattr(value, "stream"):
        raw = value.stream.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return raw

    raise TypeError(f"Unsupported text type: {type(value)!r}")


def _merge_context(task: dict, blueprint: str, overview: str) -> str:
    task_lines = ", ".join(f"{k}: {v}" for k, v in task.items())

    return (
        "```\n"
        "<TASK>\n"
        f"{task_lines}\n"
        "</TASK>\n"
        "```\n\n"
        "```\n"
        "<BLUEPRINT>\n"
        f"{blueprint}\n"
        "</BLUEPRINT>\n"
        "```\n\n"
        "```\n"
        "<OVERVIEW>\n"
        f"{overview}\n"
        "</OVERVIEW>\n"
        "```"
    )

# def to_toon(payload: Dict[str, Any] | str) -> str:
#     toon = encode(payload)
#     return f"```\n{toon}\n```"

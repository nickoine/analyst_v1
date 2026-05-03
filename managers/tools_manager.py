# Built-in
from __future__ import annotations
from typing import Any, Callable, get_args, get_origin, get_type_hints
import inspect, json, typing

# Internal
from managers.dto import ActionSpec


_JSON_TYPE_BY_PYTHON_TYPE: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _unwrap_callable(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Return the underlying function object.

    Why this exists:
    - For plain functions, `fn` is already the real function.
    - For bound methods, Python wraps the class function into a bound method.
      In that case, the original function is available via `fn.__func__`.

    This helper lets the manager consistently read:
    - decorator markers like `__agent_action__`
    - optional custom schema like `__tool_schema__`
    - docstrings
    - type hints

    while still executing the original bound callable later.
    """
    return getattr(fn, "__func__", fn)


def _json_type_from_annotation(annotation: Any) -> str:
    """
    Map a resolved Python annotation to a JSON Schema primitive type.

    Supported mappings:
    - str   -> string
    - int   -> integer
    - float -> number
    - bool  -> boolean

    Unknown annotations fall back to `"string"`.
    """
    return _JSON_TYPE_BY_PYTHON_TYPE.get(annotation, "string")


def _is_literal_annotation(annotation: Any) -> bool:
    """
    Return True if the annotation is `typing.Literal[...]`.

    A direct `get_origin(annotation) is Literal` check can be brittle across
    typing variants and Python versions, so this helper uses a slightly more
    tolerant check.
    """
    origin = get_origin(annotation)
    return origin is typing.Literal or str(origin) == "typing.Literal"


def _json_type_from_literal_values(values: tuple[Any, ...]) -> str:
    """
    Infer the JSON Schema primitive type for a `Literal[...]` annotation.

    Examples:
    - Literal["task", "blueprint"] -> "string"
    - Literal[1, 2, 3]             -> "integer"
    - Literal[1.5, 2.5]            -> "number"
    - Literal[True, False]         -> "boolean"

    Fallback is `"string"`.
    """
    if not values:
        return "string"

    if all(isinstance(v, bool) for v in values):
        return "boolean"

    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "integer"

    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return "number"

    return "string"


def _parameter_schema(param: inspect.Parameter, annotation: Any) -> dict[str, Any]:
    """
    Build JSON Schema for a single action parameter.

    Supported cases:
    - primitive annotations like `str`, `int`, `float`, `bool`
    - `Literal[...]`, which is converted into an enum
    - `list[T]`
    - `list[Literal[...]]`, which is converted into an array of enum values
    """
    schema: dict[str, Any] = {
        "description": f"Argument: {param.name}",
    }

    origin = get_origin(annotation)

    if typing.get_origin(origin) is list or origin is list:
        item_annotation = get_args(annotation)[0] if get_args(annotation) else str

        items_schema: dict[str, Any] = {}
        if _is_literal_annotation(item_annotation):
            values = get_args(item_annotation)
            items_schema["type"] = _json_type_from_literal_values(values)
            items_schema["enum"] = list(values)
        else:
            items_schema["type"] = _json_type_from_annotation(item_annotation)

        schema["type"] = "array"
        schema["items"] = items_schema
        return schema

    if _is_literal_annotation(annotation):
        values = get_args(annotation)
        schema["type"] = _json_type_from_literal_values(values)
        schema["enum"] = list(values)
        return schema

    schema["type"] = _json_type_from_annotation(annotation)
    return schema


def _iter_action_parameters(fn: Callable[..., Any]) -> list[inspect.Parameter]:
    """
    Return only user-facing parameters for an action.

    Excludes:
    - `self`
    - `cls`

    Rejects:
    - `*args`
    - `**kwargs`

    Why:
    tool schemas must expose a stable named-argument contract to the model.
    Variadic parameters are too ambiguous for that.
    """
    raw_fn = _unwrap_callable(fn)
    parameters: list[inspect.Parameter] = []

    for param in inspect.signature(raw_fn).parameters.values():
        if param.name in {"self", "cls"}:
            continue

        if param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise TypeError(
                f"{raw_fn.__name__}: *args and **kwargs are not supported for agent actions"
            )

        parameters.append(param)

    return parameters


def build_function_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """
    Build a JSON Schema parameters object from a function signature.

    This is the default automatic schema generation path used when an action
    does not provide a custom `__tool_schema__`.

    Important details:
    - Bound methods are unwrapped to their raw function first.
    - `get_type_hints()` is used instead of raw `inspect` annotations so that
      postponed annotations and `Literal[...]` work correctly.
    - Parameters without default values become required.
    - `additionalProperties` is disabled by default.

    Example result:
        {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["task", "blueprint"],
                    "description": "Argument: target"
                }
            },
            "required": ["target"],
            "additionalProperties": False
        }
    """
    raw_fn = _unwrap_callable(fn)
    parameters = _iter_action_parameters(raw_fn)
    type_hints = get_type_hints(raw_fn)

    properties = {
        param.name: _parameter_schema(
            param=param,
            annotation=type_hints.get(param.name, str),
        )
        for param in parameters
    }

    required = [
        param.name
        for param in parameters
        if param.default is inspect.Parameter.empty
    ]

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class ActionManager:
    """
    Discover, register, expose, and execute model-visible actions.

    Core responsibilities:
    1. Inspect a runtime object and discover methods marked with `@agent_action()`.
    2. Convert those methods into `ActionSpec` objects.
    3. Expose only the subset of actions allowed by the current agent profile.
    4. Execute actions by name when the model returns tool calls.

    Important design rule:
    - The runtime may define many decorated actions.
    - The model should see only those listed in `profile.actions`.
    """

    def __init__(self, runtime: Any) -> None:
        """
        Initialize the manager with a runtime object.

        Example:
            runtime = IDEData(ctx)
            manager = ActionManager(runtime)

        Args:
            runtime:
                An object whose bound methods may be exposed as model actions.
        """
        self.runtime = runtime
        self._actions: dict[str, ActionSpec] = {}

    def load_actions(self) -> None:
        """
        Discover and register bound methods marked with `@agent_action()`.

        Flow:
        1. Clear the current registry.
        2. Inspect all callable attributes on the runtime object.
        3. Unwrap bound methods to access the original function metadata.
        4. Keep only those methods marked with `__agent_action__`.
        5. Build and register an `ActionSpec` for each one.

        Result:
            self._actions becomes a registry like:
                {
                    "read_input_data": ActionSpec(...),
                    "add_numbers": ActionSpec(...),
                }
        """
        self._actions.clear()

        for _, method in inspect.getmembers(self.runtime, predicate=callable):
            raw_fn = _unwrap_callable(method)

            if getattr(raw_fn, "__agent_action__", False):
                self.register(self._build_action_spec(method))

    def register(self, action_spec: ActionSpec) -> None:
        """
        Register one action in the internal registry.

        Validation:
        - handler must be callable
        - action name must be unique
        """
        if not callable(action_spec.handler):
            raise TypeError(f"Action handler is not callable: {action_spec.name}")

        if action_spec.name in self._actions:
            raise ValueError(f"Duplicate action name: {action_spec.name}")

        self._actions[action_spec.name] = action_spec

    def get(self, name: str) -> ActionSpec | None:
        """
        Return a registered action by name, or None if not found.
        """
        return self._actions.get(name)

    def list_names(self) -> list[str]:
        """
        Return all registered action names.
        """
        return list(self._actions.keys())

    def validate_profile_actions(self, allowed_actions: list[str]) -> None:
        """
        Validate that all action names referenced by the agent profile exist.

        Raises:
            ValueError:
                If the profile references one or more unknown actions.
        """
        missing = [name for name in allowed_actions if name not in self._actions]
        if missing:
            raise ValueError(f"Unknown actions in profile: {missing}")

    def get_tool_definitions(self, allowed_actions: list[str]) -> list[dict[str, Any]]:
        """
        Build tool definitions for the subset of actions allowed by a profile.

        This is the key filter between:
        - all discovered actions in the runtime
        - only the actions the current agent is allowed to use

        Args:
            allowed_actions:
                Action names from the current agent profile.

        Returns:
            A list of tool definitions ready to be passed to the LLM client.
        """
        self.validate_profile_actions(allowed_actions)
        return [self._actions[name].as_openai_tool() for name in allowed_actions]

    def execute(self, name: str, arguments_json: str | None) -> str:
        """
        Execute one registered action by name.

        Flow:
        1. Resolve the action from the registry.
        2. Parse JSON arguments returned by the model.
        3. Call the bound Python handler with `**kwargs`.
        4. Normalize the result into a string suitable for a tool message.

        Return rules:
        - if the action returns a string -> return as is
        - otherwise -> convert the result to JSON text
        """
        action_spec = self.get(name)
        if action_spec is None:
            raise ValueError(f"Unknown action: {name}")

        kwargs = json.loads(arguments_json) if arguments_json else {}
        result = action_spec.handler(**kwargs)

        if isinstance(result, str):
            return result

        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _build_action_spec(fn: Callable[..., Any]) -> ActionSpec:
        """
        Build `ActionSpec` for a discovered action.

        Resolution order:
        1. If the underlying function defines `__tool_schema__`, use it as the
           explicit source of truth.
        2. Otherwise, build the action definition automatically from:
           - function name
           - docstring
           - generated JSON Schema from the signature

        Important:
        - Metadata such as `__tool_schema__` usually lives on the raw function.
        - Execution must still use the original callable `fn`, because for bound
          methods that preserves the attached instance (`self`).
        """
        raw_fn = _unwrap_callable(fn)
        custom_schema = getattr(raw_fn, "__tool_schema__", None)

        if custom_schema is not None:
            return ActionSpec(
                name=custom_schema["name"],
                description=custom_schema["description"],
                parameters=custom_schema["parameters"],
                strict=custom_schema.get("strict", True),
                handler=fn,
            )

        description = inspect.getdoc(raw_fn) or f"Execute action '{raw_fn.__name__}'"

        return ActionSpec(
            name=raw_fn.__name__,
            description=description,
            parameters=build_function_schema(raw_fn),
            strict=True,
            handler=fn,
        )

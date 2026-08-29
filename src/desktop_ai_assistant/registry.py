"""Declarative registration and deterministic validation of allowed tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

from .types import Primitive, ToolArguments, ToolCall, ToolResult


class ToolValidationError(ValueError):
    """Raised when untrusted model data fails a tool's schema."""


class ToolExecutionError(RuntimeError):
    """Raised when a registered tool cannot complete its work."""


@dataclass(frozen=True, slots=True)
class ArgumentSpec:
    name: str
    kind: type[str] | type[int] | type[bool]
    description: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    description: str
    arguments: tuple[ArgumentSpec, ...]


class ToolHandler(Protocol):
    def __call__(self, arguments: ToolArguments) -> str: ...


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    schema: ToolSchema
    handler: ToolHandler


class ToolRegistry:
    """The only authority for model-visible tools."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self, name: str, description: str, arguments: Sequence[ArgumentSpec] = ()
    ) -> Callable[[ToolHandler], ToolHandler]:
        schema = ToolSchema(name, description, tuple(arguments))

        def decorator(handler: ToolHandler) -> ToolHandler:
            if name in self._tools:
                raise ValueError(f"Tool already registered: {name}")
            self._tools[name] = RegisteredTool(schema, handler)
            return handler

        return decorator

    def schemas(self) -> tuple[ToolSchema, ...]:
        return tuple(tool.schema for tool in self._tools.values())

    def dispatch(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call.id, f"Unknown tool: {call.name}", is_error=True)
        try:
            validated = self._validate(tool.schema.arguments, call.arguments)
            return ToolResult(call.id, tool.handler(validated))
        except (ToolValidationError, ToolExecutionError) as error:
            return ToolResult(call.id, str(error), is_error=True)
        except Exception:
            return ToolResult(call.id, "Tool failed unexpectedly.", is_error=True)

    @staticmethod
    def _validate(
        specifications: Sequence[ArgumentSpec], arguments: ToolArguments
    ) -> ToolArguments:
        allowed_names = {specification.name for specification in specifications}
        unexpected = set(arguments).difference(allowed_names)
        if unexpected:
            raise ToolValidationError(
                f"Unexpected argument(s): {', '.join(sorted(unexpected))}"
            )
        validated: dict[str, Primitive] = {}
        for specification in specifications:
            value = arguments.get(specification.name)
            if value is None:
                if specification.required:
                    raise ToolValidationError(
                        f"Missing required argument: {specification.name}"
                    )
                continue
            if type(value) is not specification.kind:
                raise ToolValidationError(
                    f"Argument {specification.name} must be "
                    f"{specification.kind.__name__}."
                )
            validated[specification.name] = value
        return validated


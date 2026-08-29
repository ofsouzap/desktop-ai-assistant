# Desktop AI Assistant

A security-conscious, typed Python foundation for a personal Linux/Sway desktop
assistant. Milestone 1 provides a persistent REPL, provider-independent model
and tool interfaces, strict tool validation, bounded sequential orchestration,
XDG paths, structured logs, and a scripted model for deterministic testing.

It deliberately contains **no** live model provider, filesystem tool, shell
tool, inventory implementation, or Sway integration yet.

## Development

Requires Python 3.11 or newer.

```sh
python -m pip install -e ".[dev]"
pytest
mypy
```

Run the Milestone 1 mock REPL:

```sh
desktop-ai-assistant
```

The REPL keeps an in-memory conversation for its process lifetime. The current
scripted backend has no configured responses; it exists to verify the
architecture and support deterministic tests. Logs are written as JSON Lines
under `$XDG_STATE_HOME/desktop-ai-assistant/logs` (or the standard XDG
fallback).

## Security boundary

Models are untrusted planners. Only registered tools can run, and all
model-provided arguments are deterministically validated before a tool handler
receives them. There is no generic shell, Python evaluation, filesystem, or
Sway-command execution surface.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete staged project plan.

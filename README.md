# Desktop AI Assistant

A security-conscious, typed Python foundation for a personal Linux/Sway desktop
assistant. It provides a persistent REPL, provider-independent model and tool
interfaces, strict tool validation, bounded sequential orchestration, XDG paths,
structured logs, and an OpenRouter inference-only model adapter.

It deliberately contains no filesystem tool, shell tool, or Sway integration.

## Development

Requires Python 3.11 or newer.

```sh
python -m pip install -e ".[dev]"
pytest
mypy
```

Set an OpenRouter API key and optionally choose a currently available free model
that supports tools:

```sh
export OPENROUTER_API_KEY="..."
export OPENROUTER_MODEL="google/gemma-4-31b-it:free"
```

`OPENROUTER_MODEL` defaults to `google/gemma-4-31b-it:free`. Free model
availability changes; use the OpenRouter model catalog to select a free model
whose `supported_parameters` contains `tools`.

For development, enable error and traceback output in the REPL with:

```sh
DESKTOP_AI_ASSISTANT_CONSOLE_LOGS=1 desktop-ai-assistant
```

```sh
desktop-ai-assistant
```

The REPL uses OpenRouter only for inference and native tool-call generation. It
does not provide the model a local shell, filesystem, or agent runtime. The REPL
keeps an in-memory conversation for its process lifetime. Logs are written as JSON Lines under
`$XDG_STATE_HOME/desktop-ai-assistant/logs` (or the standard XDG fallback).

## Security boundary

Models are untrusted planners. Only registered tools can run, and all
model-provided arguments are deterministically validated before a tool handler
receives them. There is no generic shell, Python evaluation, filesystem, or
Sway-command execution surface.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete staged project plan.

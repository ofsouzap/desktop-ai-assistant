# Desktop AI Assistant

A security-conscious, typed Python foundation for a personal Linux/Sway desktop
assistant. It provides a persistent REPL, provider-independent model and tool
interfaces, strict tool validation, bounded sequential orchestration, XDG paths,
structured logs, and an OpenAI Codex subscription-backed model adapter.

It deliberately contains no filesystem tool, shell tool, or Sway integration.

## Development

Requires Python 3.11 or newer.

```sh
python -m pip install -e ".[dev]"
pytest
mypy
```

Authenticate once with a ChatGPT subscription:

```sh
desktop-ai-assistant login
```

Then run the REPL:

```sh
desktop-ai-assistant
```

The login command first reuses an existing Codex session; otherwise, it shows a
device-code URL and code. The Codex runtime retains the resulting session
locally; no API key is required. The REPL keeps an in-memory conversation for
its process lifetime. Logs are written as JSON Lines under
`$XDG_STATE_HOME/desktop-ai-assistant/logs` (or the standard XDG fallback).

## Security boundary

Models are untrusted planners. Only registered tools can run, and all
model-provided arguments are deterministically validated before a tool handler
receives them. There is no generic shell, Python evaluation, filesystem, or
Sway-command execution surface.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete staged project plan.

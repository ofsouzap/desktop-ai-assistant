# Desktop AI Assistant

A security-conscious, typed Python foundation for a personal Linux/Sway desktop
assistant. It provides a persistent REPL, provider-independent model and tool
interfaces, strict tool validation, bounded sequential orchestration, XDG paths,
structured logs, an OpenRouter inference-only model adapter, and constrained Sway
window/workspace tools.

It deliberately contains no filesystem tool, shell tool, or arbitrary Sway command
tool. Sway actions use concrete window IDs and named workspaces.

## Integrations

Integrations connect the assistant to narrowly defined capabilities while keeping
their data and side effects behind validated, registered tools. Each integration
must have automated tests for its implementation and behavioral evaluation cases
for the expected tool behavior and failure cases. When an integration is added or
edited, update both its automated tests and its evaluation coverage.

Integrations implement the shared interface in `desktop_ai_assistant.integrations`:
they provide `register(registry)` and an `integration_prompt` string. The prompt
is appended to the model's system prompt only when tool descriptions need extra
guidance; it may be empty.

### Inventory

The inventory integration stores UTF-8 plain text at an application-controlled
path, with one item or note per line. It can:

- read the complete inventory;
- append one non-empty inventory entry; and
- overwrite the complete inventory.

### Sway

The Sway integration uses the constrained `swaymsg` adapter and concrete window
IDs or named workspaces. It can:

- list open windows and their IDs, applications, titles, workspaces, and focus;
- list workspaces and their focus, visibility, and urgency;
- identify the currently focused window;
- focus a window by ID;
- move a window by ID to a named workspace;
- focus a named workspace; and
- enable or disable fullscreen for a window by ID.

It does not expose arbitrary Sway command execution.

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

## Behavioral evaluations

Run the deterministic inventory and mocked-Sway checks without credentials:

```sh
python -m evaluation --backend scripted --output /tmp/evaluation-traces.json
```

To evaluate the configured OpenRouter model against the same scenarios, use
`--backend openrouter`. The command writes complete JSON traces, including the prompt,
conversation, tool calls, response, errors, and objective checks. A non-zero
exit status means at least one objective check failed; qualitative `REVIEW`
results should be inspected rather than reduced to a single score.

When handing a trace to a coding agent, provide the JSON file and ask it to
review each scenario's tool sequence, arguments, normalized tool results,
final response, and objective checks separately. Distinguish model-planning
failures from tool/API or orchestration failures before changing code.

For development, enable error and traceback output in the REPL with:

```sh
DESKTOP_AI_ASSISTANT_CONSOLE_LOGS=1 desktop-ai-assistant
```

```sh
desktop-ai-assistant
```

The REPL uses OpenRouter only for inference and native tool-call generation. It
does not provide the model a local shell, filesystem, arbitrary Sway command, or
agent runtime. The REPL keeps an in-memory conversation for its process lifetime.
Logs are written as JSON Lines under
`$XDG_STATE_HOME/desktop-ai-assistant/logs` (or the standard XDG fallback).

## Security boundary

Models are untrusted planners. Only registered tools can run, and all
model-provided arguments are deterministically validated before a tool handler
receives them. There is no generic shell, Python evaluation, filesystem, or
Sway-command execution surface.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete staged project plan.

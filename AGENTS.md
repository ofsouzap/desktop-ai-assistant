# AGENTS.md

## Development environment

- Use `uv` for the project environment and dependency management.
- Use Python 3.11 or newer.
- Install/sync development dependencies with:
  `uv sync --extra dev`
- Run project commands through `uv run`; do not rely on globally installed
  Python packages.

## Validation

Before submitting changes, run:

```sh
uv run pytest -q
uv run mypy .
```

For the deterministic behavioral evaluations, run:

```sh
uv run python -m evaluation --backend scripted --output /tmp/evaluation-traces.json
```

Use the OpenRouter evaluation backend only when credentials are configured.

## Project conventions

- Application code belongs under `src/desktop_ai_assistant/`.
- Add or update focused tests under `tests/` for behavior changes.
- Keep strict typing enabled and preserve the provider-independent interfaces.
- Treat model output and tool arguments as untrusted input. Validate every
  tool call in Python before it reaches an integration or causes an effect.
- Do not add arbitrary shell, Python evaluation, filesystem, web browsing, or
  generic Sway-command execution capabilities without an explicit design
  change.
- Respect the existing XDG path helpers and structured logging conventions.
- Keep changes focused, avoid unrelated reformatting, and update `README.md`
  or `PROJECT_PLAN.md` when development or architectural behavior changes.

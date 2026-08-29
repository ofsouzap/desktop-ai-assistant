# Desktop AI Assistant — V1 Project Plan

## Project goal

Build a personal AI-powered desktop assistant for Linux/Sway. V1 provides a
persistent text REPL that translates natural-language requests into a small,
explicit set of structured tool calls. It is intended for responsive, short
desktop tasks rather than long-running autonomous work.

## Core security model

- Treat the LLM as an untrusted planner: it requests tools but never directly
  executes code or desktop actions.
- Only registered tools run automatically; all arguments are deterministically
  validated in Python.
- Do not provide arbitrary shell execution, `eval`/`exec`, generic Sway
  commands, arbitrary filesystem access, web browsing, screenshots, or
  arbitrary input simulation.
- Treat tool-returned content as untrusted data.
- Container/rootless isolation and confirmation tiers are future hardening
  work, not V1 requirements.

## Architecture

- Persistent CLI/REPL with concise responses and visible tool calls.
- In-memory conversation session for the REPL lifetime.
- Bounded sequential orchestrator which calls a provider-independent model
  backend, dispatches validated calls through a central registry, feeds results
  back, and ends on a final response.
- Typed model, message, tool-call, result, error, and future Sway-data
  representations.
- Declarative tool registry, XDG path helper, structured logging, inventory
  storage adapter, and Sway adapter boundary.
- Provider/backend implementations remain replaceable and suitable for future
  rootless-container separation.

## Milestones

### 1. Repository, typed architecture, CI, and mock orchestration

Implement the Python package skeleton; reproducible development setup;
persistent REPL; strict static type checking; provider-independent types and
model interface; declarative registry, schemas, validation, and dispatch;
bounded sequential loop (initially five calls); in-memory session context; XDG
paths; full-fidelity structured logging; scripted model backend; recoverable
error handling; deterministic tests; and GitHub Actions running tests and
strict typing. Do not add real providers, inventory effects, or Sway effects.
Stop for human architectural review.

### 2. Inventory tools and safe local persistence

Add fixed-XDG-path inventory read, append, and atomic overwrite tools. Keep
human-readable free-form entries, document the format in write-tool
descriptions, impose reasonable limits, log recoverable mutation data, and add
mocked deterministic tests.

### 3. Strong real model backend

First investigate hosted model choices and stop for human selection. Then
implement exactly one provider behind the model interface, test mocked provider
responses, and keep live calls out of CI.

### 4. Sway desktop integration

Add a typed `swaymsg` adapter with known-safe structured invocations and
normalized errors/data. Add list/focused window/workspace, focus, move, and
fullscreen tools without generic command text. Test subprocess handling with
mocks and manually audit real-Sway writes.

### 5. Behavioral evaluation suite and reliability pass

Build a small manually run evaluation harness using mocked inventory and Sway
state. Retain JSON traces, objectively check useful expectations, and use
human/agent review for qualitative assessment.

### 6. V1 polish, installability, and security review

Review validation, tool descriptions, logging, error paths, REPL ergonomics,
and XDG behavior. Provide install/run documentation, keep CI green, execute
evaluations, and perform a final manual security review.

## Deferred goals

Rootless isolation, confirmation policies, parallel calls, richer desktop
operations, screenshot/general computer use, persistent conversation or
instructions, other integrations, local models, and voice are explicitly
post-V1. Reassess the security model before every major capability expansion.

.PHONY: check

check:
	uv run mypy . && \
	uv run pytest . && \
	uv run ruff check

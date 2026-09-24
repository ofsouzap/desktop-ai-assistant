.PHONY: check

check:
	uv run mypy . && \
	uv run ruff check && \
	uv run pytest .

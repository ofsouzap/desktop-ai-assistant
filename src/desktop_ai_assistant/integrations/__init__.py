"""External integrations for the desktop assistant."""

from typing import Protocol

from ..registry import ToolRegistry


class Integration(Protocol):
	"""A capability that can register tools and optional model guidance."""

	@property
	def integration_prompt(self) -> str:
		"""Return optional guidance to append to the system prompt."""
		...

	def register(self, registry: ToolRegistry) -> None:
		"""Register every tool provided by this integration."""

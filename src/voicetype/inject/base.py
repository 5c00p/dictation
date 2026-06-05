"""Text injector interface."""

from __future__ import annotations

import abc


class TextInjector(abc.ABC):
    @abc.abstractmethod
    def inject(self, text: str) -> None:
        """Insert ``text`` at the current caret position in the active window."""

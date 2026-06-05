"""Text injection into the active input field."""

from __future__ import annotations

from ..config import InjectConfig
from .base import TextInjector


def create_injector(cfg: InjectConfig) -> TextInjector:
    if cfg.method == "unicode":
        from .unicode_input import UnicodeInjector

        return UnicodeInjector()
    from .clipboard import ClipboardInjector

    return ClipboardInjector(
        restore=cfg.restore_clipboard,
        pre_delay=cfg.paste_pre_delay,
        post_delay=cfg.paste_post_delay,
    )


__all__ = ["TextInjector", "create_injector"]

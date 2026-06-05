"""Primary injection method: clipboard paste (spec 7.1).

Save the current clipboard -> put recognized text -> emulate Ctrl+V -> restore.
This is the most reliable way to insert Unicode (Cyrillic/Latin) into Electron
apps, browsers and 1C:EDT.

Nuances handled (spec 7.1):
* Restore the clipboard with a delay, otherwise the target app may paste the old
  value.
* If the clipboard held non-text content (e.g. an image), we cannot faithfully
  save/restore it here — we log a warning and skip restoration rather than
  clobbering it with text.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

from .base import TextInjector

logger = logging.getLogger(__name__)

_WIN = sys.platform == "win32"

if _WIN:
    import win32clipboard
    import win32con


def _open_clipboard(retries: int = 10, delay: float = 0.02) -> bool:
    for _ in range(retries):
        try:
            win32clipboard.OpenClipboard()
            return True
        except Exception:
            time.sleep(delay)
    return False


def _get_clipboard_text() -> tuple[Optional[str], bool]:
    """Return ``(text, had_nontext)``.

    ``text`` is the current Unicode clipboard text or ``None``; ``had_nontext``
    is True when the clipboard held some non-text format we won't restore.
    """
    if not _WIN or not _open_clipboard():
        return None, False
    try:
        if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
            return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT), False
        # Something is on the clipboard but it isn't text.
        try:
            had = bool(win32clipboard.EnumClipboardFormats(0))
        except Exception:
            had = False
        return None, had
    except Exception as exc:  # pragma: no cover
        logger.debug("Reading clipboard failed: %s", exc)
        return None, False
    finally:
        win32clipboard.CloseClipboard()


def _set_clipboard_text(text: str) -> bool:
    if not _WIN or not _open_clipboard():
        return False
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
        return True
    except Exception as exc:  # pragma: no cover
        logger.debug("Writing clipboard failed: %s", exc)
        return False
    finally:
        win32clipboard.CloseClipboard()


class ClipboardInjector(TextInjector):
    def __init__(
        self,
        restore: bool = True,
        pre_delay: float = 0.05,
        post_delay: float = 0.12,
    ) -> None:
        self.restore = restore
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        from pynput.keyboard import Controller, Key, KeyCode

        self._kb = Controller()
        self._Key = Key
        # Press 'V' by virtual-key code (VK_V = 0x56), NOT by the character "v".
        # Under a non-Latin layout (e.g. Russian, which is active while dictating)
        # the char "v" does not map to VK_V, so Ctrl+"v" would not trigger paste.
        # The vk is layout-independent and always means the physical V key.
        self._v_key = KeyCode.from_vk(0x56)

    def _send_paste(self) -> None:
        Key = self._Key
        self._kb.press(Key.ctrl)
        self._kb.press(self._v_key)
        self._kb.release(self._v_key)
        self._kb.release(Key.ctrl)

    def inject(self, text: str) -> None:
        if not text:
            return
        if not _WIN:
            logger.error("Clipboard injection is only implemented on Windows")
            return

        prev_text, had_nontext = (None, False)
        if self.restore:
            prev_text, had_nontext = _get_clipboard_text()
            if had_nontext:
                logger.warning(
                    "Clipboard held non-text content; it will not be restored"
                )

        if not _set_clipboard_text(text):
            logger.error("Could not set clipboard; aborting paste")
            return

        time.sleep(self.pre_delay)
        self._send_paste()
        time.sleep(self.post_delay)

        if self.restore and prev_text is not None:
            _set_clipboard_text(prev_text)

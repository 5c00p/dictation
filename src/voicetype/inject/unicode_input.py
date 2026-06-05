"""Fallback injection: per-character Unicode SendInput (spec 7.2).

Sends each character via ``SendInput`` with ``KEYEVENTF_UNICODE`` so it works in
fields where Ctrl+V is intercepted by the application. Handles characters outside
the BMP via surrogate pairs.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from ctypes import wintypes

from .base import TextInjector

logger = logging.getLogger(__name__)

_WIN = sys.platform == "win32"

KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1


if _WIN:
    ULONG_PTR = ctypes.POINTER(wintypes.ULONG)

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class _INPUTunion(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("union", _INPUTunion)]

    def _make_input(scan: int, key_up: bool) -> "INPUT":
        flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if key_up else 0)
        ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=None)
        return INPUT(type=INPUT_KEYBOARD, union=_INPUTunion(ki=ki))


class UnicodeInjector(TextInjector):
    def __init__(self, char_delay: float = 0.0) -> None:
        self.char_delay = char_delay

    def inject(self, text: str) -> None:
        if not text or not _WIN:
            if text and not _WIN:
                logger.error("Unicode SendInput is only implemented on Windows")
            return

        inputs: list[INPUT] = []
        for ch in text:
            # Build down+up for each UTF-16 code unit (surrogate pairs included).
            for code_unit in _utf16_units(ch):
                inputs.append(_make_input(code_unit, key_up=False))
                inputs.append(_make_input(code_unit, key_up=True))

        if not inputs:
            return

        n = len(inputs)
        arr = (INPUT * n)(*inputs)
        sent = ctypes.windll.user32.SendInput(n, arr, ctypes.sizeof(INPUT))
        if sent != n:
            logger.warning("SendInput sent %d of %d events", sent, n)
        if self.char_delay:
            time.sleep(self.char_delay)


def _utf16_units(ch: str) -> list[int]:
    """Return the UTF-16 code units (one, or a surrogate pair) for a char."""
    data = ch.encode("utf-16-le")
    return [int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2)]

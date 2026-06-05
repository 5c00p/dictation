"""Global hotkey handling via pynput.

pynput's ``GlobalHotKeys`` only fires on activation, but push-to-talk needs both
press *and* release of a chord (hold to talk, release to stop). So we run a raw
``keyboard.Listener`` that tracks the set of currently-pressed keys and derives
edge events ourselves:

* push-to-talk: fire ``on_ptt_start`` when the chord becomes fully held and
  ``on_ptt_stop`` when it stops being fully held.
* toggle / switch-language: fire once on the press edge.

Hotkey strings look like ``"ctrl+alt"`` or ``"ctrl+alt+space"``. Modifier names
(``ctrl``/``alt``/``shift``/``win``) match either left or right physical key.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from pynput import keyboard

logger = logging.getLogger(__name__)

# Map config modifier names to the set of pynput Keys that satisfy them.
_MODIFIER_ALIASES: dict[str, set] = {
    "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
    "control": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
    "alt": {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr},
    "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
    "win": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
    "cmd": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
    "super": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
}

_NAMED_KEYS: dict[str, object] = {
    "space": keyboard.Key.space,
    "enter": keyboard.Key.enter,
    "tab": keyboard.Key.tab,
    "esc": keyboard.Key.esc,
    "escape": keyboard.Key.esc,
    "f1": keyboard.Key.f1, "f2": keyboard.Key.f2, "f3": keyboard.Key.f3,
    "f4": keyboard.Key.f4, "f5": keyboard.Key.f5, "f6": keyboard.Key.f6,
    "f7": keyboard.Key.f7, "f8": keyboard.Key.f8, "f9": keyboard.Key.f9,
    "f10": keyboard.Key.f10, "f11": keyboard.Key.f11, "f12": keyboard.Key.f12,
}


class _Token:
    """One required key of a chord: either a modifier-alias set or a single key."""

    def __init__(self, name: str) -> None:
        self.name = name
        if name in _MODIFIER_ALIASES:
            self.options = set(_MODIFIER_ALIASES[name])
            self.char: Optional[str] = None
        elif name in _NAMED_KEYS:
            self.options = {_NAMED_KEYS[name]}
            self.char = None
        elif len(name) == 1:
            self.options = set()
            self.char = name
        else:
            logger.warning("Unknown hotkey token %r; it will never match", name)
            self.options = set()
            self.char = None

    def matches(self, pressed_keys: set, pressed_chars: set) -> bool:
        if self.char is not None:
            return self.char in pressed_chars
        return bool(self.options & pressed_keys)


def _parse_chord(spec: str) -> list[_Token]:
    return [_Token(part.strip().lower()) for part in spec.split("+") if part.strip()]


class HotkeyManager:
    def __init__(
        self,
        push_to_talk: str,
        toggle: str,
        switch_language: str,
        on_ptt_start: Callable[[], None],
        on_ptt_stop: Callable[[], None],
        on_toggle: Callable[[], None],
        on_switch_language: Callable[[], None],
    ) -> None:
        self._ptt = _parse_chord(push_to_talk)
        self._toggle = _parse_chord(toggle)
        self._switch = _parse_chord(switch_language)

        self._on_ptt_start = on_ptt_start
        self._on_ptt_stop = on_ptt_stop
        self._on_toggle = on_toggle
        self._on_switch = on_switch_language

        self._pressed_keys: set = set()
        self._pressed_chars: set = set()
        self._ptt_active = False
        self._toggle_held = False
        self._switch_held = False

        self._listener: Optional[keyboard.Listener] = None

    # -- chord evaluation -------------------------------------------------
    def _chord_held(self, chord: list[_Token]) -> bool:
        if not chord:
            return False
        return all(tok.matches(self._pressed_keys, self._pressed_chars) for tok in chord)

    def _evaluate(self) -> None:
        # Push-to-talk: track held state, fire on both edges.
        ptt_now = self._chord_held(self._ptt)
        if ptt_now and not self._ptt_active:
            self._ptt_active = True
            self._safe(self._on_ptt_start)
        elif not ptt_now and self._ptt_active:
            self._ptt_active = False
            self._safe(self._on_ptt_stop)

        # Toggle: fire once per press edge.
        toggle_now = self._chord_held(self._toggle)
        if toggle_now and not self._toggle_held:
            self._toggle_held = True
            self._safe(self._on_toggle)
        elif not toggle_now:
            self._toggle_held = False

        # Switch language: fire once per press edge.
        switch_now = self._chord_held(self._switch)
        if switch_now and not self._switch_held:
            self._switch_held = True
            self._safe(self._on_switch)
        elif not switch_now:
            self._switch_held = False

    @staticmethod
    def _safe(cb: Callable[[], None]) -> None:
        try:
            cb()
        except Exception as exc:  # never let a callback kill the listener
            logger.exception("Hotkey callback failed: %s", exc)

    # -- listener plumbing ------------------------------------------------
    @staticmethod
    def _char_of(key) -> Optional[str]:  # noqa: ANN001
        try:
            if isinstance(key, keyboard.KeyCode) and key.char:
                return key.char.lower()
        except Exception:
            return None
        return None

    def _on_press(self, key) -> None:  # noqa: ANN001
        self._pressed_keys.add(key)
        ch = self._char_of(key)
        if ch:
            self._pressed_chars.add(ch)
        self._evaluate()

    def _on_release(self, key) -> None:  # noqa: ANN001
        self._pressed_keys.discard(key)
        ch = self._char_of(key)
        if ch:
            self._pressed_chars.discard(ch)
        self._evaluate()

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()
        logger.info(
            "Hotkeys active — PTT=%s toggle=%s switch=%s",
            "+".join(t.name for t in self._ptt),
            "+".join(t.name for t in self._toggle),
            "+".join(t.name for t in self._switch),
        )

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

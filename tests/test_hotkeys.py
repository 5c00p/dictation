"""Tests for hotkey chord parsing and edge detection (no real keyboard)."""

from pynput import keyboard

from voicetype.hotkeys import HotkeyManager, _parse_chord


def _make_manager():
    events = {"ptt_start": 0, "ptt_stop": 0, "toggle": 0, "switch": 0}
    mgr = HotkeyManager(
        push_to_talk="ctrl+alt",
        toggle="ctrl+alt+space",
        switch_language="ctrl+alt+l",
        on_ptt_start=lambda: events.__setitem__("ptt_start", events["ptt_start"] + 1),
        on_ptt_stop=lambda: events.__setitem__("ptt_stop", events["ptt_stop"] + 1),
        on_toggle=lambda: events.__setitem__("toggle", events["toggle"] + 1),
        on_switch_language=lambda: events.__setitem__("switch", events["switch"] + 1),
    )
    return mgr, events


def test_parse_chord():
    chord = _parse_chord("ctrl+alt+space")
    assert [t.name for t in chord] == ["ctrl", "alt", "space"]


def test_ptt_fires_on_both_edges():
    mgr, events = _make_manager()
    mgr._on_press(keyboard.Key.ctrl_l)
    assert events["ptt_start"] == 0  # only ctrl so far
    mgr._on_press(keyboard.Key.alt_l)
    assert events["ptt_start"] == 1  # chord complete
    assert events["ptt_stop"] == 0
    mgr._on_release(keyboard.Key.alt_l)
    assert events["ptt_stop"] == 1
    # No spurious extra starts.
    assert events["ptt_start"] == 1


def test_toggle_fires_once_per_press():
    mgr, events = _make_manager()
    mgr._on_press(keyboard.Key.ctrl_l)
    mgr._on_press(keyboard.Key.alt_l)
    mgr._on_press(keyboard.Key.space)
    assert events["toggle"] == 1
    # Holding doesn't refire.
    mgr._on_press(keyboard.Key.space)
    assert events["toggle"] == 1
    mgr._on_release(keyboard.Key.space)
    mgr._on_release(keyboard.Key.alt_l)
    mgr._on_release(keyboard.Key.ctrl_l)
    # Press again -> fires again.
    mgr._on_press(keyboard.Key.ctrl_l)
    mgr._on_press(keyboard.Key.alt_l)
    mgr._on_press(keyboard.Key.space)
    assert events["toggle"] == 2


def test_switch_language_char_key():
    mgr, events = _make_manager()
    mgr._on_press(keyboard.Key.ctrl_l)
    mgr._on_press(keyboard.Key.alt_l)
    mgr._on_press(keyboard.KeyCode.from_char("l"))
    assert events["switch"] == 1

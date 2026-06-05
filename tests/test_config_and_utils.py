"""Config loading and small utility tests."""

import sys

from voicetype.config import Config, load_config


def test_default_config():
    cfg = Config()
    assert cfg.stt.provider == "local_whisper"
    assert cfg.stt.default_language == "ru"
    assert cfg.hotkeys.push_to_talk == "ctrl+alt"
    assert cfg.inject.method == "clipboard"
    assert cfg.app.log_transcripts is False


def test_load_config_from_toml(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[stt]\nmodel = "small"\ndefault_language = "be"\n'
        '[app]\nlog_level = "DEBUG"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.stt.model == "small"
    assert cfg.stt.default_language == "be"
    assert cfg.app.log_level == "DEBUG"
    # Untouched sections keep defaults.
    assert cfg.inject.method == "clipboard"


def test_load_config_invalid_falls_back(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("this is : not valid toml = [", encoding="utf-8")
    cfg = load_config(p)
    assert isinstance(cfg, Config)  # fell back to defaults, didn't raise


def test_utf16_units_bmp_and_surrogate():
    if sys.platform != "win32":
        # The module guards ctypes structures behind win32; the helper itself is
        # pure-Python and importable everywhere.
        pass
    from voicetype.inject.unicode_input import _utf16_units

    assert _utf16_units("A") == [0x0041]
    assert _utf16_units("я") == [0x044F]
    # An emoji outside the BMP becomes a surrogate pair.
    pair = _utf16_units("😀")
    assert len(pair) == 2
    assert 0xD800 <= pair[0] <= 0xDBFF
    assert 0xDC00 <= pair[1] <= 0xDFFF

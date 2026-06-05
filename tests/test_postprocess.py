"""Tests for text postprocessing (no audio / model needed)."""

from voicetype.config import PostprocessConfig
from voicetype.postprocess import postprocess


def _cfg(**kw):
    base = dict(
        capitalize_first=True,
        voice_punctuation=True,
        numbers_to_digits=False,
        raw_mode=False,
    )
    base.update(kw)
    return PostprocessConfig(**base)


def test_capitalize_first():
    assert postprocess("привет мир", _cfg(voice_punctuation=False)) == "Привет мир"


def test_voice_punctuation_attached():
    # "точка" attaches to the previous word: no space before the dot.
    out = postprocess("привет мир точка", _cfg())
    assert out == "Привет мир."


def test_voice_punctuation_comma_and_question():
    out = postprocess("да запятая правда знак вопроса", _cfg())
    assert out == "Да, правда?"


def test_voice_punctuation_newline():
    out = postprocess("первая строка новая строка вторая", _cfg())
    assert out == "Первая строка\nвторая"


def test_belarusian_punctuation():
    out = postprocess("прывітанне свет кропка", _cfg())
    assert out == "Прывітанне свет."


def test_raw_mode_bypasses_everything():
    text = "привет мир точка"
    assert postprocess(text, _cfg(raw_mode=True)) == text


def test_numbers_to_digits():
    out = postprocess(
        "у меня два кота",
        _cfg(voice_punctuation=False, numbers_to_digits=True),
    )
    assert out == "У меня 2 кота"


def test_numbers_off_by_default():
    out = postprocess("у меня два кота", _cfg(voice_punctuation=False))
    assert out == "У меня два кота"


def test_empty_string():
    assert postprocess("", _cfg()) == ""

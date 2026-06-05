"""Text postprocessing: voice punctuation, capitalization, numbers (spec 3.2).

All steps are individually toggleable in config. ``raw_mode`` bypasses
everything (useful for dictating code or logs). Supports RU and BE command words.
"""

from __future__ import annotations

import re

from .config import PostprocessConfig

# Voice punctuation commands -> inserted token. The boolean marks whether the
# token "attaches" to the preceding word (no leading space) like ``.`` / ``,``.
_PUNCT_COMMANDS: dict[str, tuple[str, bool]] = {
    # Russian
    "точка": (".", True),
    "запятая": (",", True),
    "вопросительный знак": ("?", True),
    "знак вопроса": ("?", True),
    "восклицательный знак": ("!", True),
    "двоеточие": (":", True),
    "точка с запятой": (";", True),
    "тире": (" —", False),
    "дефис": ("-", True),
    "открыть скобку": (" (", False),
    "закрыть скобку": (")", True),
    "кавычки": ('"', False),
    "новая строка": ("\n", False),
    "новый абзац": ("\n\n", False),
    "абзац": ("\n\n", False),
    # Belarusian
    "кропка": (".", True),
    "коска": (",", True),
    "клічнік": ("!", True),
    "пытальнік": ("?", True),
    "двукроп'е": (":", True),
    "новы радок": ("\n", False),
    "новы абзац": ("\n\n", False),
}

# Number words -> digit, RU + BE (0..20 and tens; enough for common dictation).
_NUMBER_WORDS: dict[str, str] = {
    "ноль": "0", "нуль": "0",
    "один": "1", "одна": "1", "адзін": "1", "адна": "1",
    "два": "2", "две": "2", "дзве": "2",
    "три": "3", "тры": "3",
    "четыре": "4", "чатыры": "4",
    "пять": "5", "пяць": "5",
    "шесть": "6", "шэсць": "6",
    "семь": "7", "сем": "7",
    "восемь": "8", "восем": "8",
    "девять": "9", "дзевяць": "9",
    "десять": "10", "дзесяць": "10",
    "одиннадцать": "11", "адзінаццаць": "11",
    "двенадцать": "12", "дванаццаць": "12",
    "тринадцать": "13", "трынаццаць": "13",
    "четырнадцать": "14", "чатырнаццаць": "14",
    "пятнадцать": "15", "пятнаццаць": "15",
    "шестнадцать": "16", "шаснаццаць": "16",
    "семнадцать": "17", "сямнаццаць": "17",
    "восемнадцать": "18", "васямнаццаць": "18",
    "девятнадцать": "19", "дзевятнаццаць": "19",
    "двадцать": "20", "дваццаць": "20",
    "тридцать": "30", "трыццаць": "30",
    "сорок": "40", "сорак": "40",
    "пятьдесят": "50", "пяцьдзесят": "50",
    "сто": "100", "тысяча": "1000", "тысяч": "1000",
}

# Sentinel marking punctuation that should swallow the whitespace before it.
_ATTACH = "\x00"


def _apply_voice_punctuation(text: str) -> str:
    result = text
    # Longer phrases first so "знак вопроса" wins over a hypothetical "знак".
    for phrase in sorted(_PUNCT_COMMANDS, key=len, reverse=True):
        token, attached = _PUNCT_COMMANDS[phrase]
        pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
        replacement = (_ATTACH + token) if attached else token
        result = pattern.sub(lambda m, r=replacement: r, result)
    # Attached punctuation: drop spaces/tabs before the sentinel, then the sentinel.
    result = re.sub(r"[ \t]*\x00", "", result)
    # Trim spaces/tabs that ended up directly around an inserted newline.
    result = re.sub(r"[ \t]*\n[ \t]*", "\n", result)
    return result


def _apply_numbers(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        word = m.group(0)
        digit = _NUMBER_WORDS.get(word.lower())
        return digit if digit is not None else word

    return re.sub(r"\w+", repl, text)


def _capitalize_first(text: str) -> str:
    stripped = text.lstrip()
    if not stripped:
        return text
    lead = text[: len(text) - len(stripped)]
    return lead + stripped[0].upper() + stripped[1:]


def postprocess(text: str, cfg: PostprocessConfig) -> str:
    if not text or cfg.raw_mode:
        return text

    result = text
    if cfg.voice_punctuation:
        result = _apply_voice_punctuation(result)
    if cfg.numbers_to_digits:
        result = _apply_numbers(result)
    # Tidy up whitespace introduced by replacements.
    result = re.sub(r"[ \t]{2,}", " ", result).strip()
    if cfg.capitalize_first:
        result = _capitalize_first(result)
    return result

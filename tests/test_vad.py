"""Tests for the energy-based voice segmenter."""

import numpy as np

from voicetype.audio.vad import VoiceSegmenter, _rms


def _frame(amplitude: float, n: int = 512) -> np.ndarray:
    # Deterministic "speech-like" frame: a sine at the given amplitude.
    t = np.arange(n) / 16000.0
    return (amplitude * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_rms_zero_for_silence():
    assert _rms(np.zeros(512, dtype=np.float32)) == 0.0


def test_segment_emitted_after_pause():
    seg = VoiceSegmenter(
        sample_rate=16000,
        frame_ms=32,
        min_silence_ms=96,   # 3 frames
        min_speech_ms=64,    # 2 frames
        speech_pad_ms=64,
    )
    # Prime noise floor with quiet frames.
    for _ in range(5):
        assert seg.feed(_frame(0.0005)) is None
    # Speech frames (none should emit yet).
    for _ in range(6):
        assert seg.feed(_frame(0.3)) is None
    # Silence -> should emit the buffered utterance.
    result = None
    for _ in range(5):
        r = seg.feed(_frame(0.0))
        if r is not None:
            result = r
    assert result is not None
    assert result.size > 0


def test_short_blip_discarded():
    seg = VoiceSegmenter(
        sample_rate=16000,
        frame_ms=32,
        min_silence_ms=96,
        min_speech_ms=320,   # require 10 frames of speech
        speech_pad_ms=0,
    )
    for _ in range(5):
        seg.feed(_frame(0.0005))
    # Only 2 speech frames, then silence -> below min_speech -> discarded.
    seg.feed(_frame(0.3))
    seg.feed(_frame(0.3))
    result = None
    for _ in range(5):
        result = result or seg.feed(_frame(0.0))
    assert result is None


def test_flush_returns_trailing_speech():
    seg = VoiceSegmenter(
        sample_rate=16000,
        frame_ms=32,
        min_silence_ms=320,
        min_speech_ms=64,
        speech_pad_ms=0,
    )
    for _ in range(5):
        seg.feed(_frame(0.0005))
    for _ in range(6):
        seg.feed(_frame(0.3))
    # No long pause; flush should hand back the in-progress utterance.
    tail = seg.flush()
    assert tail is not None and tail.size > 0

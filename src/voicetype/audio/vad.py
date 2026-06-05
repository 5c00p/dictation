"""Voice-activity detection for the toggle (continuous) dictation mode.

Implements an online, energy-based segmenter that splits the incoming audio
stream into utterances separated by pauses (spec 7.3 — "chunked transcription by
VAD"). It is intentionally model-free so it adds zero warm-up latency and runs
trivially on CPU.

The class keeps a short pre-roll so the leading phoneme is not clipped, tracks
an adaptive noise floor (so it adapts to room/mic gain), and emits a completed
segment once a configurable trailing silence is observed. It deliberately mirrors
the silero-vad parameter names (``threshold``, ``min_silence_ms``,
``min_speech_ms``, ``speech_pad_ms``) so a silero implementation can be swapped in
later without touching the controller.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


def _rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


class VoiceSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 32,
        threshold: float = 0.5,
        min_silence_ms: int = 700,
        min_speech_ms: int = 250,
        speech_pad_ms: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        # ``threshold`` in [0, 1] scales how far above the noise floor counts as
        # speech. We map it to an energy multiplier: higher -> stricter.
        self._sensitivity = 1.0 + 4.0 * float(threshold)

        self._min_silence_frames = max(1, min_silence_ms // frame_ms)
        self._min_speech_frames = max(1, min_speech_ms // frame_ms)
        self._pad_frames = max(0, speech_pad_ms // frame_ms)

        self._noise_rms = 1e-3  # adaptive noise floor, seeded low
        self._in_speech = False
        self._speech_frames: list[np.ndarray] = []
        self._silence_run = 0
        self._speech_run = 0
        self._preroll: deque[np.ndarray] = deque(maxlen=self._pad_frames or 1)

    def reset(self) -> None:
        self._in_speech = False
        self._speech_frames = []
        self._silence_run = 0
        self._speech_run = 0
        self._preroll.clear()

    def _is_speech(self, energy: float) -> bool:
        return energy > self._noise_rms * self._sensitivity

    def feed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process one frame; return a finished utterance if a pause ended one."""
        energy = _rms(frame)

        if not self._in_speech:
            # Adapt noise floor only while idle (exponential moving average).
            self._noise_rms = 0.95 * self._noise_rms + 0.05 * energy
            self._preroll.append(frame)
            if self._is_speech(energy):
                self._speech_run += 1
                if self._speech_run >= 1:
                    # Enter speech; prepend the pre-roll padding.
                    self._in_speech = True
                    self._speech_frames = list(self._preroll)
                    self._silence_run = 0
            else:
                self._speech_run = 0
            return None

        # In speech.
        self._speech_frames.append(frame)
        if self._is_speech(energy):
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._min_silence_frames:
                return self._finish()
        return None

    def _finish(self) -> Optional[np.ndarray]:
        frames = self._speech_frames
        self.reset()
        if len(frames) < self._min_speech_frames:
            return None  # too short — likely a noise blip
        return np.concatenate(frames).astype(np.float32)

    def flush(self) -> Optional[np.ndarray]:
        """Return any buffered speech when the stream stops (hotkey toggled off)."""
        if not self._in_speech:
            return None
        return self._finish()

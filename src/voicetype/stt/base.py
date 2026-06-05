"""Abstract STT provider interface.

The whole point of this abstraction (spec section 5) is that the engine —
local Whisper, OpenAI, Yandex, … — can be swapped without touching capture,
injection or the controller. A provider only has to turn a mono float32 numpy
array into text.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None


class STTProvider(abc.ABC):
    @abc.abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe a complete utterance (mono float32, 16 kHz)."""

    def transcribe_stream(
        self,
        chunks: Iterable[np.ndarray],
        language: Optional[str] = None,
    ) -> Iterator[TranscriptionResult]:
        """Transcribe a sequence of VAD-segmented chunks.

        Default implementation just maps :meth:`transcribe` over the chunks;
        providers with true streaming APIs can override this.
        """
        for chunk in chunks:
            result = self.transcribe(chunk, language=language)
            if result.text:
                yield result

    def warmup(self) -> None:  # pragma: no cover - optional hook
        """Optionally pre-load weights / run a dummy inference at startup."""

    def close(self) -> None:  # pragma: no cover - optional hook
        """Release any resources."""

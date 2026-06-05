"""Microphone capture via sounddevice.

Provides two capture styles used by the controller:

* :class:`PushToTalkRecorder` — start/stop recording bounded by a hotkey hold,
  returning the whole utterance as one float32 array.
* :class:`StreamingRecorder` — a background stream that pushes fixed-size frames
  to a queue, consumed by the VAD segmenter for toggle (continuous) mode.

All audio is mono float32 in [-1, 1] at the configured sample rate (16 kHz by
default, which is what Whisper expects).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


def resolve_device(name: str) -> Optional[int]:
    """Resolve a (sub)string device name to a sounddevice input index.

    Empty string -> ``None`` (system default). Unknown name -> ``None`` with a
    warning, so a misconfigured device never crashes startup.
    """
    if not name:
        return None
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and name.lower() in dev["name"].lower():
                return idx
    except Exception as exc:  # pragma: no cover - depends on host audio stack
        logger.warning("Could not enumerate audio devices: %s", exc)
    logger.warning("Input device %r not found; using system default", name)
    return None


def list_input_devices() -> list[tuple[int, str]]:
    """Return ``(index, name)`` for all devices with input channels."""
    devices: list[tuple[int, str]] = []
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append((idx, dev["name"]))
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not enumerate audio devices: %s", exc)
    return devices


class PushToTalkRecorder:
    """Records audio while active and returns the full buffer on stop."""

    def __init__(self, sample_rate: int = 16000, device: Optional[int] = None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._frames: list[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("Input stream status: %s", status)
        with self._lock:
            self._frames.append(indata[:, 0].copy())

    def start(self) -> None:
        with self._lock:
            self._frames = []
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            logger.error("Failed to start microphone stream: %s", exc)
            self._stream = None

    def stop(self) -> np.ndarray:
        """Stop the stream and return the captured mono float32 audio."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # pragma: no cover
                logger.warning("Error closing stream: %s", exc)
        with self._lock:
            frames = self._frames
            self._frames = []
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).astype(np.float32)

    @property
    def is_recording(self) -> bool:
        return self._stream is not None


class StreamingRecorder:
    """Continuous capture feeding fixed-size frames into a queue.

    Frames are ``frame_ms`` long (default 32 ms). The consumer (VAD segmenter)
    pulls them off :attr:`frames`.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: Optional[int] = None,
        frame_ms: int = 32,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frames: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("Input stream status: %s", status)
        self.frames.put(indata[:, 0].copy())

    def start(self) -> None:
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
                blocksize=self.frame_samples,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            logger.error("Failed to start streaming microphone: %s", exc)
            self._stream = None

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # pragma: no cover
                logger.warning("Error closing stream: %s", exc)
        # Drain any pending frames.
        with self.frames.mutex:
            self.frames.queue.clear()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

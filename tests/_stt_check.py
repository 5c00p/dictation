"""Manual end-to-end STT check on a real (TTS-generated) WAV using the tiny model.

Not part of the pytest suite (underscore prefix) — downloads the tiny model and
exercises the actual faster-whisper path on CPU.
"""

import wave

import numpy as np

from voicetype.config import STTConfig
from voicetype.stt.local_whisper import LocalWhisperProvider


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return pcm


cfg = STTConfig(model="tiny", device="cpu", compute_type="int8")
provider = LocalWhisperProvider(cfg)
audio = load_wav("tests/_sample.wav")
print(f"audio: {audio.size} samples, {audio.size / 16000:.2f}s")
result = provider.transcribe(audio, language="en")
print("LANG:", result.language)
print("TEXT:", repr(result.text))
assert result.text.strip(), "expected non-empty transcription"
assert "test" in result.text.lower(), "expected the word 'test' in the result"
print("STT END-TO-END OK")

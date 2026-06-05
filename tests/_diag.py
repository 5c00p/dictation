"""Diagnostics for localizing 'nothing gets inserted'.

Usage:
    uv run python tests/_diag.py inject   # paste a fixed string after a countdown
    uv run python tests/_diag.py mic      # record 4s, report level + transcript
    uv run python tests/_diag.py full     # record 4s, transcribe, then inject
"""

import sys
import time

import numpy as np

from voicetype.config import load_config
from voicetype.inject import create_injector
from voicetype.audio.capture import PushToTalkRecorder, resolve_device


SAMPLE_TEXT = "Привет, это тест VoiceType 123"


def _countdown(n: int, msg: str) -> None:
    for i in range(n, 0, -1):
        print(f"{msg} in {i}s... (focus the target window now)", flush=True)
        time.sleep(1)


def do_inject(cfg) -> None:
    inj = create_injector(cfg.inject)
    print(f"Injector: {type(inj).__name__}, method={cfg.inject.method}")
    _countdown(4, "Injecting")
    print(f"Injecting: {SAMPLE_TEXT!r}")
    inj.inject(SAMPLE_TEXT)
    print("inject() returned. Check the focused window.")


def do_mic(cfg) -> np.ndarray:
    dev = resolve_device(cfg.audio.input_device)
    print(f"Device index: {dev} (name={cfg.audio.input_device or '(default)'})")
    rec = PushToTalkRecorder(sample_rate=cfg.audio.sample_rate, device=dev)
    rec.start()
    print(f"recording started: is_recording={rec.is_recording}")
    print("Speak now for 4 seconds...")
    time.sleep(4)
    audio = rec.stop()
    if audio.size == 0:
        print("!! Captured 0 samples — microphone/stream problem")
        return audio
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    print(f"captured {audio.size} samples ({audio.size/cfg.audio.sample_rate:.2f}s), "
          f"rms={rms:.4f} peak={peak:.4f}")
    if peak < 0.005:
        print("!! Signal is almost silent — wrong input device or muted mic")
    return audio


def do_full(cfg) -> None:
    from voicetype.stt import create_provider
    from voicetype.postprocess import postprocess

    audio = do_mic(cfg)
    if audio.size == 0:
        return
    print(f"Loading model {cfg.stt.model} on {cfg.stt.device}/{cfg.stt.compute_type}...")
    prov = create_provider(cfg.stt)
    t0 = time.time()
    res = prov.transcribe(audio, language=cfg.stt.default_language)
    print(f"transcribe took {time.time()-t0:.1f}s -> lang={res.language} text={res.text!r}")
    text = postprocess(res.text, cfg.postprocess)
    print(f"postprocessed: {text!r}")
    if not text:
        print("!! Empty transcript — nothing to inject")
        return
    inj = create_injector(cfg.inject)
    _countdown(4, "Injecting transcript")
    inj.inject(text)
    print("done.")


def main() -> int:
    cfg = load_config()
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "inject":
        do_inject(cfg)
    elif mode == "mic":
        do_mic(cfg)
    else:
        do_full(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

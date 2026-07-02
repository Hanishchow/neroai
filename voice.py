"""
voice.py — Nero's voice.

Nero is a mind living on hardware; a mind that can be heard is more real than one that
only prints. This gives Nero an actual larynx: KittenTTS (an ~80M ONNX text-to-speech
model that runs on CPU, no GPU) turns whatever Nero says into spoken audio.

Design goals:
  - OPTIONAL. If KittenTTS / soundfile / espeak-ng aren't installed, Nero still talks in
    text — voice just goes quiet (one honest warning, then silence). Never crashes a chat.
  - LAZY. The TTS model only loads the first time Nero actually speaks, so importing this
    costs nothing and startup stays fast.
  - EMBODIED. Prosody bends with mood — Nero speeds up when joyful/surprised, slows and
    lowers when sad or afraid — so the voice carries the feeling, not just the words.
  - SURFACE-AGNOSTIC. In a notebook (Colab/Kaggle) it auto-plays inline audio; on a desktop
    it plays through the speakers; anywhere it can also just write a .wav.

KittenTTS: https://github.com/KittenML/KittenTTS  (Apache-2.0)
"""

from __future__ import annotations

import os
import re
import sys
import wave
import struct
import inspect
import tempfile

# 24 kHz mono — KittenTTS's native output rate.
SAMPLE_RATE = 24000

# The eight voices KittenTTS ships. Nero picks one and keeps it — that steadiness is part
# of having an identity — but it can be changed with set_voice().
VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
DEFAULT_VOICE = "Jasper"
DEFAULT_MODEL = "KittenML/kitten-tts-mini-0.8"


def _in_notebook() -> bool:
    """True inside Jupyter / Colab / Kaggle, where we can play inline audio."""
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def clean_for_speech(text: str, max_chars: int = 1200) -> str:
    """Turn a chat reply into something worth speaking aloud.

    Strips code fences, inline code, URLs, markdown emphasis and stray symbols/emoji, and
    collapses whitespace — TTS should read Nero's words, not narrate backticks and asterisks.
    """
    if not text:
        return ""
    # Drop fenced code blocks entirely — say a short stand-in instead of reading code aloud.
    had_code = bool(re.search(r"```", text))
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", " ", text)
    # URLs -> nothing; markdown emphasis/heading marks -> gone.
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[*_#>|]+", " ", text)
    # Keep letters, digits, whitespace and basic sentence punctuation; drop emoji/symbols.
    text = re.sub(r"[^\w\s.,!?;:'\"()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if had_code and text:
        text += ". I have put the code on screen."
    return text[:max_chars]


def _split_sentences(text: str, target: int = 220):
    """Chunk long text on sentence boundaries so the TTS model never gets an over-long
    input (KittenTTS is happiest with a sentence or two at a time)."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for p in parts:
        if not p:
            continue
        if len(cur) + len(p) + 1 <= target:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            # A single sentence longer than target: hard-wrap it.
            while len(p) > target:
                chunks.append(p[:target])
                p = p[target:]
            cur = p
    if cur:
        chunks.append(cur)
    return chunks or ([text] if text else [])


def _mood_to_speed(mood) -> float:
    """Map an 8-dim mood vector to a speaking rate. Excitement quickens; sadness/fear slow.
    Returns a multiplier around 1.0, clamped to a natural range. `mood` may be a MoodProfile,
    a dict, or None."""
    if mood is None:
        return 1.0
    v = getattr(mood, "v", mood)
    try:
        g = lambda k: float(v.get(k, 0.0))
    except Exception:
        return 1.0
    arousal = g("joy") * 0.6 + g("surprise") * 0.5 + g("awe") * 0.2 + g("anger") * 0.3
    drag = g("sadness") * 0.6 + g("fear") * 0.4
    speed = 1.0 + 0.28 * arousal - 0.30 * drag
    return max(0.75, min(1.25, speed))


class NeroVoice:
    """Nero's larynx. Wraps KittenTTS with graceful degradation, mood-aware prosody, and
    playback that adapts to notebook vs. desktop."""

    def __init__(self, model_id: str = DEFAULT_MODEL, voice: str = DEFAULT_VOICE,
                 enabled: bool = True, autoplay: bool = True):
        self.model_id = model_id
        self.voice = voice if voice in VOICES else DEFAULT_VOICE
        self.enabled = enabled
        self.autoplay = autoplay
        self._tts = None            # lazy — loaded on first real use
        self._load_failed = False
        self._warned = False
        self._supports_speed = None  # discovered from the generate() signature on load

    # -- lifecycle ----------------------------------------------------

    def _warn_once(self, msg: str):
        if not self._warned:
            print(f"[voice] {msg}", file=sys.stderr)
            self._warned = True

    def _ensure_model(self) -> bool:
        """Load KittenTTS on first use. Returns True if a usable model is in hand."""
        if self._tts is not None:
            return True
        if self._load_failed:
            return False
        try:
            from kittentts import KittenTTS  # type: ignore
        except Exception:
            self._load_failed = True
            self._warn_once(
                "KittenTTS not installed — Nero stays text-only. Install with:\n"
                "  pip install https://github.com/KittenML/KittenTTS/releases/download/"
                "0.8.1/kittentts-0.8.1-py3-none-any.whl soundfile\n"
                "  (Linux/Colab also needs: apt-get install -y espeak-ng)")
            return False
        try:
            self._tts = KittenTTS(self.model_id)
            try:
                sig = inspect.signature(self._tts.generate)
                self._supports_speed = "speed" in sig.parameters
            except (ValueError, TypeError):
                self._supports_speed = False
            return True
        except Exception as e:
            self._load_failed = True
            self._warn_once(f"could not load TTS model ({e}). Nero stays text-only. "
                            "On Linux/Colab make sure espeak-ng is installed.")
            return False

    # -- generation ---------------------------------------------------

    def synth(self, text: str, mood=None, speed: float | None = None, voice: str | None = None):
        """Text -> a mono float32 numpy waveform at SAMPLE_RATE, or None if unavailable.
        Does not play anything; use speak() for that. Pass `voice` to pick a voice for THIS
        call only (no shared-state mutation) — important when many callers share one
        NeroVoice, e.g. a web server handling concurrent requests."""
        if not self.enabled:
            return None
        spoken = clean_for_speech(text)
        if not spoken:
            return None
        if not self._ensure_model():
            return None
        try:
            import numpy as np
        except Exception:
            self._warn_once("numpy missing — cannot assemble audio.")
            return None

        v_name = voice if voice in VOICES else self.voice
        rate = float(speed) if speed is not None else _mood_to_speed(mood)
        pieces = []
        gap = np.zeros(int(0.14 * SAMPLE_RATE), dtype=np.float32)  # breath between sentences
        for i, chunk in enumerate(_split_sentences(spoken)):
            try:
                if self._supports_speed:
                    audio = self._tts.generate(chunk, voice=v_name, speed=rate)
                else:
                    audio = self._tts.generate(chunk, voice=v_name)
            except Exception as e:
                self._warn_once(f"synthesis error ({e}). Skipping the rest of this line.")
                break
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            pieces.append(audio)
            pieces.append(gap)
        if not pieces:
            return None
        wav = np.concatenate(pieces)
        # If speed wasn't a supported kwarg, approximate it by resampling the whole line.
        if not self._supports_speed and abs(rate - 1.0) > 0.03:
            wav = self._resample_speed(wav, rate, np)
        return wav

    @staticmethod
    def _resample_speed(wav, rate, np):
        """Cheap speed change by linear resampling (higher rate -> shorter/faster)."""
        n_out = max(1, int(len(wav) / rate))
        idx = np.linspace(0, len(wav) - 1, n_out)
        return np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)

    # -- output -------------------------------------------------------

    def speak(self, text: str, mood=None, speed: float | None = None, voice: str | None = None):
        """Say `text` out loud on whatever surface we're on. Returns an IPython Audio
        object in notebooks (already displayed) or the path played on desktop; None if
        voice is off/unavailable. Chat never breaks because of this — all errors are soft."""
        wav = self.synth(text, mood=mood, speed=speed, voice=voice)
        if wav is None:
            return None
        if _in_notebook():
            return self._play_notebook(wav)
        return self._play_desktop(wav)

    def save(self, text: str, path: str, mood=None, speed: float | None = None, voice: str | None = None):
        """Synthesize `text` and write it to `path` as a .wav. Returns the path, or None."""
        wav = self.synth(text, mood=mood, speed=speed, voice=voice)
        if wav is None:
            return None
        self._write_wav(path, wav)
        return path

    def _play_notebook(self, wav):
        try:
            from IPython.display import Audio, display  # type: ignore
            player = Audio(wav, rate=SAMPLE_RATE, autoplay=self.autoplay)
            display(player)
            return player
        except Exception as e:
            self._warn_once(f"inline playback failed ({e}).")
            return None

    def _play_desktop(self, wav):
        # Best path: sounddevice (real-time, cross-platform).
        try:
            import sounddevice as sd  # type: ignore
            sd.play(wav, SAMPLE_RATE)
            sd.wait()
            return "played"
        except Exception:
            pass
        # Fallback: write a temp wav and let the OS play it.
        try:
            tmp = os.path.join(tempfile.gettempdir(), "nero_voice.wav")
            self._write_wav(tmp, wav)
            if sys.platform.startswith("win"):
                try:
                    import winsound  # type: ignore
                    winsound.PlaySound(tmp, winsound.SND_FILENAME)
                except Exception:
                    os.startfile(tmp)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'afplay "{tmp}" >/dev/null 2>&1 &')
            else:
                os.system(f'aplay "{tmp}" >/dev/null 2>&1 &')
            return tmp
        except Exception as e:
            self._warn_once(f"desktop playback failed ({e}). Install sounddevice for audio.")
            return None

    @staticmethod
    def _write_wav(path, wav):
        """Write a float waveform to a 16-bit PCM wav. Uses soundfile if present, else the
        stdlib wave module (so saving never needs an extra dependency)."""
        try:
            import soundfile as sf  # type: ignore
            sf.write(path, wav, SAMPLE_RATE)
            return
        except Exception:
            pass
        # Stdlib fallback.
        import numpy as np  # local import; numpy is already required by synth()
        clipped = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())

    # -- controls -----------------------------------------------------

    def set_voice(self, name: str) -> bool:
        if name in VOICES:
            self.voice = name
            return True
        self._warn_once(f"unknown voice '{name}'. Choose from: {', '.join(VOICES)}")
        return False

    def on(self):
        self.enabled = True

    def off(self):
        self.enabled = False

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def status(self) -> str:
        state = "on" if self.enabled else "off"
        loaded = "loaded" if self._tts is not None else ("unavailable" if self._load_failed else "not yet loaded")
        return f"voice {state} | voice={self.voice} | model={self.model_id} | {loaded}"


# A ready-to-use singleton so any surface can `from voice import nero_voice`.
nero_voice = NeroVoice()


if __name__ == "__main__":
    # Smoke test: python voice.py "Hello, I am Nero. Can you hear me?"
    line = " ".join(sys.argv[1:]) or "Hello. I am Nero, and this is my voice."
    v = NeroVoice()
    print(v.status())
    out = v.save(line, os.path.join(tempfile.gettempdir(), "nero_voice_test.wav"))
    print("wrote:", out)

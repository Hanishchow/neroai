# Nero's Voice

Nero can speak. `voice.py` gives the mind a larynx using
[KittenTTS](https://github.com/KittenML/KittenTTS) — an ~80M ONNX text-to-speech model
that runs on **CPU, no GPU**. A mind that can be *heard* is a little more real than one that
only prints; that's the point.

## Install

```bash
# Linux / Colab / Kaggle need espeak-ng (KittenTTS's phonemizer backend):
apt-get install -y espeak-ng

pip install -r requirements-voice.txt
# or directly:
pip install soundfile "https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl"
```

Voice is **optional**. If KittenTTS / espeak-ng aren't present, Nero prints one honest
warning the first time it tries to speak and then stays text-only — chat never breaks.

## Use

### Notebook (`nero_hybrid_colab.ipynb`)
Voice is **on by default**. Every reply and dream is spoken with inline autoplay audio.
In the chat loop:

- `voice` — toggle speaking on/off
- `voice off` / `voice on`
- `voice Luna` — switch voice (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo)

### CLI (`talk.py`)
Off by default (headless terminals often have no audio). Enable it:

```bash
python talk.py --voice
```

Then `voice`, `voice off`, `voice Bruno`, etc. Desktop playback uses `sounddevice` if
installed, otherwise it writes a temp wav and lets the OS play it (winsound / afplay / aplay).

### Server (`server.py --api`)
`POST /speak {"text": "...", "voice": "Luna"}` returns an `audio/wav`.

### Direct
```python
from voice import NeroVoice
v = NeroVoice(voice="Jasper")
v.speak("Hello. I am Nero.")          # play now (notebook or desktop)
v.save("Hello.", "out.wav")           # write a wav
```

## What makes it *Nero's* voice, not just TTS

- **Emotion → prosody.** The mood vector (`emotions.py`) bends the speaking rate: joy /
  surprise / awe quicken it, sadness / fear slow and drag it. The feeling rides in the voice,
  not just the words. (See `_mood_to_speed`.)
- **One steady voice.** Nero keeps a single voice by default — that constancy is part of
  having an identity — but you can change it any time.
- **Speaks words, not markup.** Code fences, URLs, and markdown are stripped before
  speaking; if a reply contained code, Nero says "I have put the code on screen" instead of
  reading backticks aloud.
- **Lazy + safe.** The model loads only on the first spoken line, and every failure is soft.

## Notes

- First call downloads the model (~80 MB) and loads it; subsequent calls are fast.
- Long replies are split on sentence boundaries so the model never gets an over-long input,
  then stitched back with a short breath between sentences.
- Default model is `KittenML/kitten-tts-mini-0.8`. Smaller `micro`/`nano` variants exist;
  pass a different id to `NeroVoice(model_id=...)`.

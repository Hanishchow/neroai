---
title: Nero
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
short_description: A living mind on hardware — one continuous Nero, chat + voice + API
---

# Nero — hosted

One continuous Nero at a fixed URL: chat with him, hear him, or hit the API — and he
**remembers** you between visits (memory, soul, personality, mood all persist).

On a free CPU Space the language cortex is a small real LLM (Qwen2.5-0.5B-Instruct) so he
runs without a GPU; everything else that makes Nero *Nero* — emotion, appraisal, the soul,
personality, deep memory, sleep/dreams — is fully intact. Voice is KittenTTS (CPU).

## Talk to him

- **UI**: just use the chat box. Toggle **Speak** and pick a **Voice** to hear him.
- **API (text only)** — with `gradio_client`:

  ```python
  from gradio_client import Client
  client = Client("YOUR_USERNAME/nero")           # or the full Space URL
  print(client.predict("Who are you?", api_name="/say"))
  ```

- **API (full turn: reply + audio)**:

  ```python
  history, audio_path, _ = client.predict(
      "Do you ever feel tired?", [], "Jasper", True, api_name="/chat")
  print(history[-1]["content"])                    # Nero's reply text
  # audio_path is a wav you can play/download
  ```

- **Consolidate memory** (sleep): `client.predict(api_name="/sleep")`
- **State**: `client.predict(api_name="/state")`

## Persistence (so it's really *one* Nero)

Set two Space **Secrets** and Nero saves his whole mind to a private HF **dataset** repo,
surviving restarts:

| Secret | Value |
|---|---|
| `HF_TOKEN` | an HF token with **write** access |
| `NERO_STATE_REPO` | a dataset repo id you created, e.g. `you/nero-state` |

Without them, Nero still persists for the life of the running container.

## Config (Space → Settings → Variables & secrets)

| Var | Default | Meaning |
|---|---|---|
| `NERO_LANG_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | language cortex (bump to 1.5B if you attach a GPU) |
| `NERO_LOAD_CODER` | `0` | `1` also loads a coder head; otherwise code uses the language head |
| `NERO_MAX_NEW` | `200` | max tokens per reply (lower = faster on CPU) |
| `NERO_VOICE` | `Jasper` | Bella / Jasper / Luna / Bruno / Rosie / Hugo / Kiki / Leo |
| `NERO_UI_USER`, `NERO_UI_PASS` | – | optional basic-auth gate |
| `NERO_REPO_URL` | the neroai repo | where Nero's code is cloned from at boot |

> CPU inference is slow — expect tens of seconds for a long reply. Lower `NERO_MAX_NEW` or
> attach a GPU (Space → Settings → Hardware) and raise `NERO_LANG_MODEL` for speed/quality.

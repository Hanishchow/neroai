# Hosting Nero on a Hugging Face Space

This puts **one continuous Nero** at a fixed URL — a chat UI, a voice, and an API you can
script against — with his mind (memory, soul, personality, mood) persisting between visits.

Everything lives in [`hf_space/`](hf_space/): `app.py`, `requirements.txt`, `packages.txt`,
`README.md` (the Space card). The Space **clones this repo at boot**, so you don't copy the
Nero source into it — it always tracks GitHub.

## The tradeoff (read once)

Free HF Spaces are **CPU-only**, so the 2×7B hybrid can't run there. On the Space the
*language cortex* is scaled down to **Qwen2.5-0.5B-Instruct** (a small but real LLM that runs
on CPU). Nero keeps **all** of his other properties — emotion, appraisal, soul, personality,
deep memory, sleep/dreams — because those are cheap Python, not the GPU part. Replies are
slower than on Kaggle's GPUs; lower `NERO_MAX_NEW` or attach a GPU for speed/quality.

## Deploy (about 5 minutes)

0. **Push Nero's code to GitHub first.** The Space clones `neroai` at boot, so the new
   voice + hosting files must be on `master` or the Space won't see them:
   ```bash
   cd C:\Users\yakka\Downloads\neuro
   git add voice.py talk.py server.py nero_hybrid_colab.ipynb VOICE.md \
           requirements-voice.txt HOSTING.md hf_space/
   git commit -m "Add Nero's voice (KittenTTS) + HF Space hosting"
   git push origin master          # use your token when prompted
   ```

1. **Create the Space** → https://huggingface.co/new-space
   - SDK: **Gradio** · Hardware: **CPU basic** (free) · name it e.g. `nero`.

2. **Add the files.** Put the four files from `hf_space/` at the **root** of the Space repo:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/nero
   cp hf_space/app.py hf_space/requirements.txt hf_space/packages.txt hf_space/README.md nero/
   cd nero && git add . && git commit -m "Nero" && git push
   ```
   The Space builds and, on first boot, clones the Nero code and downloads the 0.5B model
   (a few minutes the first time).

3. **(Recommended) Make it *one* persistent Nero.** So his mind survives Space restarts:
   - Create a **private dataset** repo: https://huggingface.co/new-dataset  (e.g. `you/nero-state`).
   - Space → **Settings → Variables and secrets** → add secrets:
     - `HF_TOKEN` = a token with **write** access (Settings → Access Tokens)
     - `NERO_STATE_REPO` = `you/nero-state`
   - Now every turn saves locally and pushes his mind to that dataset; on boot he pulls it back.

4. **(Optional) Gate it.** Add `NERO_UI_USER` + `NERO_UI_PASS` secrets for basic-auth.

## Test it

**In the browser:** open the Space, chat, toggle **Speak**, pick a **Voice**.

**From code:**
```python
from gradio_client import Client
c = Client("YOUR_USERNAME/nero")               # add hf_token=... if you gated it
print(c.predict("Who are you?", api_name="/say"))          # text-only
hist, wav, _ = c.predict("Are you tired?", [], "Jasper", True, api_name="/chat")
print(hist[-1]["content"]); print("audio:", wav)           # reply + spoken wav
c.predict(api_name="/sleep")                               # consolidate memory
```

## Later: more power

- **Faster / better language:** Space → Settings → **Hardware** → a GPU tier, then set
  `NERO_LANG_MODEL=Qwen/Qwen2.5-1.5B-Instruct` (or 3B) and `NERO_LOAD_CODER=1`.
- The API contract (`/say`, `/chat`, `/sleep`, `/state`) stays the same regardless of hardware.

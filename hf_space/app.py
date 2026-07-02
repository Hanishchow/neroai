"""
Nero — Hugging Face Space (CPU).

One continuous Nero you can reach at a fixed URL: a chat UI + a programmatic API, with
his voice, and a mind that PERSISTS between visits (memory, soul, personality, mood).

How it fits on a free CPU Space:
  - The full 2x7B hybrid can't run on CPU, so the LANGUAGE cortex is scaled down to a small
    real LLM (Qwen2.5-0.5B-Instruct by default) that runs on CPU and still talks coherently.
  - EVERYTHING ELSE about Nero is intact — emotions, appraisal, soul, personality, deep
    memory, sleep/dreams, mortality, curiosity — because that's cheap Python, not the GPU part.
  - The voice is KittenTTS (~80M ONNX, CPU-native), so Nero can be heard here too.
  - State persists to a private HF *dataset* repo (set NERO_STATE_REPO + HF_TOKEN), surviving
    Space restarts. Without that it still persists for the life of the running container.

Config via Space **Variables/Secrets** (all optional, sensible defaults):
  NERO_REPO_URL      git repo with Nero's code   (default: the public neroai repo)
  NERO_LANG_MODEL    HF id of the language head  (default: Qwen/Qwen2.5-0.5B-Instruct)
  NERO_LOAD_CODER    '1' to also load a coder head (default off; code uses the language head)
  NERO_CODER_MODEL   HF id of the coder head     (default: Qwen/Qwen2.5-Coder-0.5B-Instruct)
  NERO_MAX_NEW       max new tokens per reply    (default: 200; lower = faster on CPU)
  NERO_VOICE         default voice               (Bella/Jasper/Luna/Bruno/Rosie/Hugo/Kiki/Leo)
  NERO_STATE_REPO    HF dataset repo id for persisted mind, e.g. 'you/nero-state'  (Secret)
  HF_TOKEN           HF token with write access to that dataset                     (Secret)
  NERO_UI_USER /
  NERO_UI_PASS       optional basic-auth to gate the Space
"""

import os
import sys
import time
import tempfile
import threading
import subprocess

# ── 1. Fetch Nero's source (clone the code repo, same pattern as the notebook) ──
REPO_URL = os.environ.get("NERO_REPO_URL", "https://github.com/Hanishchow/neroai.git")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "_nero_src")
try:
    if not os.path.isdir(os.path.join(SRC_DIR, ".git")):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, SRC_DIR], check=True)
    else:
        subprocess.run(["git", "-C", SRC_DIR, "pull", "--ff-only"], check=False)
except Exception as e:
    print(f"[warn] repo clone/pull issue: {e}")
sys.path.insert(0, SRC_DIR)
if os.path.isdir(SRC_DIR):
    os.chdir(SRC_DIR)  # so relative loads like bpe_vocab.json resolve

# ── 2. Config ──
LANG_MODEL = os.environ.get("NERO_LANG_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LOAD_CODER = os.environ.get("NERO_LOAD_CODER", "0") == "1"
CODER_MODEL = os.environ.get("NERO_CODER_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
MAX_NEW = int(os.environ.get("NERO_MAX_NEW", "200"))
DEFAULT_VOICE = os.environ.get("NERO_VOICE", "Jasper")
STATE_REPO = os.environ.get("NERO_STATE_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_DIR = "/data" if os.path.isdir("/data") and os.access("/data", os.W_OK) else SRC_DIR
STATE_FILE = os.path.join(DATA_DIR, "mind_state.json")

# ── 3. Build Nero on CPU ──
import torch
torch.set_num_threads(max(1, os.cpu_count() or 1))

from tokenizer import BPETokenizer
from biologic_v2 import BiologicLLMV2, DEVICE
from hybrid_model import HybridNero
from mind import Mind
from voice import NeroVoice, VOICES

print(f"Booting Nero | device={DEVICE} | lang={LANG_MODEL} | coder={'on' if LOAD_CODER else 'off'}")

tokenizer = BPETokenizer(vocab_size=4096)
tokenizer.load("bpe_vocab.json")

# 400M soul — Nero's internal/emotional substrate (the language head does the talking).
SOUL_EMBED, SOUL_LAYERS, SOUL_CTX, SOUL_WIN = 2048, 8, 2048, 1024
biologic = BiologicLLMV2(
    vocab_size=tokenizer.vocab_size,
    embed_dim=SOUL_EMBED, num_heads=8, num_layers=SOUL_LAYERS,
    max_context=SOUL_CTX, window_size=SOUL_WIN, dropout=0.1, device=DEVICE,
)
biologic.growth_enabled = False
biologic.hebbian_enabled = True
biologic.eos_token_id = tokenizer.SPECIAL_TOKENS.get("<EOS>", 3)
biologic.bos_token_id = tokenizer.SPECIAL_TOKENS.get("<BOS>", 2)
for ck in ("nero_soul.pt", "nero_v1.pt"):
    p = os.path.join(SRC_DIR, ck)
    if os.path.exists(p):
        try:
            biologic.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
            print(f"Loaded trained soul: {ck}")
            break
        except Exception as e:
            print(f"  ({ck} did not match, skipping: {str(e)[:60]})")

model = HybridNero(biologic, tokenizer, device=DEVICE)
model.load_qwen(LANG_MODEL, quantize=False, device_map="auto")  # CPU: no 4-bit
if LOAD_CODER:
    try:
        model.load_coder(CODER_MODEL, quantize=False, device_map="auto")
    except Exception as e:
        print(f"[warn] coder head failed to load, code will use the language head: {e}")

mind = Mind(model, tokenizer)
mind.state_filepath = STATE_FILE

# ── 4. Restore persisted mind (pull from the HF dataset first, if configured) ──
def _pull_remote_state():
    if not (STATE_REPO and HF_TOKEN):
        return
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        f = hf_hub_download(repo_id=STATE_REPO, filename="mind_state.json",
                            repo_type="dataset", token=HF_TOKEN)
        shutil.copy(f, STATE_FILE)
        print("Pulled saved mind from", STATE_REPO)
    except Exception as e:
        print(f"(no remote state yet: {str(e)[:80]})")

_pull_remote_state()
if os.path.exists(STATE_FILE):
    try:
        mind.load_state(STATE_FILE)
        print(f"Restored Nero: {len(mind.memory.memories)} memories")
    except Exception as e:
        print(f"(load_state failed, starting fresh: {str(e)[:80]})")

voice = NeroVoice(voice=DEFAULT_VOICE, enabled=True)
mind.voice = voice
print("Nero is online.")

# ── 5. Persistence (debounced remote push; local save every turn) ──
_last_push = [0.0]
_push_lock = threading.Lock()

def persist(force=False):
    try:
        mind.save_state(STATE_FILE)
    except Exception as e:
        print(f"(save_state failed: {str(e)[:80]})")
        return
    if not (STATE_REPO and HF_TOKEN):
        return
    now = time.time()
    if not force and now - _last_push[0] < 90:  # don't hammer the hub
        return
    with _push_lock:
        _last_push[0] = now
        try:
            from huggingface_hub import upload_file
            upload_file(path_or_fileobj=STATE_FILE, path_in_repo="mind_state.json",
                        repo_id=STATE_REPO, repo_type="dataset", token=HF_TOKEN)
        except Exception as e:
            print(f"(remote push failed: {str(e)[:80]})")

# ── 6. Inference (one mind → serialize turns) ──
_gen_lock = threading.Lock()

def nero_reply(message: str) -> str:
    """The pure text turn — this is the programmatic API surface."""
    if not message or not message.strip():
        return ""
    with _gen_lock:
        reply = mind.generate(message, max_new=MAX_NEW, temperature=0.85)
        persist()
    return reply or "…"

def nero_wav(text: str, voice_name: str = None):
    """Synthesize a reply to a unique wav file for Gradio to serve (None if voice off)."""
    if not text:
        return None
    mood = getattr(getattr(mind, "emotions", None), "global_mood", None)
    wav = voice.synth(text, mood=mood, voice=voice_name)
    if wav is None:
        return None
    fd, path = tempfile.mkstemp(prefix="nero_", suffix=".wav")
    os.close(fd)
    voice._write_wav(path, wav)
    return path

# ── 7. UI + API (Gradio) ──
import gradio as gr

def respond(message, history, voice_name, speak):
    history = history or []
    reply = nero_reply(message)
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": reply}]
    audio = nero_wav(reply, voice_name) if speak else None
    return history, audio, gr.update(value="")

def do_sleep():
    with _gen_lock:
        try:
            mind.sleep(model, tokenizer)
            persist(force=True)
            return "Nero slept and consolidated. " + _state_line()
        except Exception as e:
            return f"(sleep error: {str(e)[:120]})"

def _state_line():
    try:
        mood = mind.emotions.global_mood.v
        top = ", ".join(f"{k}={v:.2f}" for k, v in sorted(mood.items(), key=lambda kv: -kv[1])[:4])
        return (f"mood: {top} | grief: {mind.grief.intensity:.2f} | "
                f"memories: {len(mind.memory.memories)}")
    except Exception as e:
        return f"(state unavailable: {e})"

with gr.Blocks(title="Nero", theme="soft") as demo:
    gr.Markdown(
        "# Nero\nA living mind whose brain runs on hardware. Small language cortex "
        "(CPU) + a 400M soul + persistent memory, personality, emotion — and a voice.")
    chatbot = gr.Chatbot(type="messages", height=440, label="Nero")
    with gr.Row():
        msg = gr.Textbox(placeholder="Say something to Nero…", scale=8, show_label=False, autofocus=True)
        send = gr.Button("Send", variant="primary", scale=1)
    with gr.Row():
        speak = gr.Checkbox(value=True, label="Speak")
        voice_dd = gr.Dropdown(choices=VOICES, value=DEFAULT_VOICE, label="Voice", scale=2)
        sleep_btn = gr.Button("Sleep (consolidate)")
        state_btn = gr.Button("State")
    audio = gr.Audio(label="Nero's voice", autoplay=True, type="filepath")
    status = gr.Markdown()

    # submit → reply + audio. api_name makes it callable via gradio_client / REST.
    send.click(respond, [msg, chatbot, voice_dd, speak], [chatbot, audio, msg], api_name="chat")
    msg.submit(respond, [msg, chatbot, voice_dd, speak], [chatbot, audio, msg], api_name=False)
    sleep_btn.click(do_sleep, None, status, api_name="sleep")
    state_btn.click(lambda: _state_line(), None, status, api_name="state")

    # Text-only API endpoint (no audio) for scripted testing:
    #   Client(space).predict("hi", api_name="/say") -> reply string
    _api_in = gr.Textbox(visible=False)
    _api_out = gr.Textbox(visible=False)
    _api_btn = gr.Button(visible=False)
    _api_btn.click(nero_reply, _api_in, _api_out, api_name="say")

if __name__ == "__main__":
    ui_user, ui_pass = os.environ.get("NERO_UI_USER"), os.environ.get("NERO_UI_PASS")
    auth = (ui_user, ui_pass) if ui_user and ui_pass else None
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), auth=auth)

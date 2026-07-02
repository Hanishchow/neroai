"""
Nero API server — run on any GPU, chat from anywhere.
Usage:
  python server.py --load nero_v1.pt              # localhost:8000
  python server.py --load nero_v1.pt --share       # public gradio URL
  python server.py --load nero_v1.pt --api         # FastAPI (curl/scripts)
"""
import sys, os, time, argparse, threading, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--load', default=None, help='Path to model checkpoint')
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--api', action='store_true', help='Enable FastAPI endpoint')
parser.add_argument('--share', action='store_true', help='Public gradio URL')
args = parser.parse_args()

# --- Init model ---
from tokenizer import BPETokenizer
tokenizer = BPETokenizer(vocab_size=4096)
tokenizer.load("bpe_vocab.json")
print(f"Vocab: {tokenizer.get_vocab_size()} tokens")

from biologic_v2 import BiologicLLMV2, DEVICE
import torch

model = BiologicLLMV2(
    vocab_size=tokenizer.vocab_size,
    embed_dim=1408, num_heads=8, num_layers=8,
    max_context=16384, window_size=1024, device=DEVICE
)
model.growth_enabled = False
model.eval()
model.eos_token_id = tokenizer.SPECIAL_TOKENS.get('<EOS>', 3)

if args.load and os.path.exists(args.load):
    sd = torch.load(args.load, map_location=DEVICE, weights_only=True)
    try:
        model.load_state_dict(sd)
        print(f"Loaded: {args.load}")
    except RuntimeError as e:
        print(f"[WARNING] Checkpoint mismatch — running with random weights.")
        print(f"  Detail: {e}")
elif args.load:
    print(f"[WARNING] Checkpoint not found: {args.load}")
    print(f"  Hint: use the full path, e.g. --load C:\\Users\\yakka\\Downloads\\nero_model\\nero_v1.pt")

from mind import Mind
mind = Mind(model, tokenizer)
print(f"Model online: {sum(p.numel() for p in model.parameters()):,} params on {DEVICE}")

# --- Gradio chat UI (easiest) ---
import gradio as gr

def chat(message, history):
    if not message:
        return ""
    reply = mind.generate(message, max_new=300, temperature=0.85)
    if not reply:
        prompt = f"User: {message}\n"
        ids = tokenizer.encode(prompt)
        if len(ids) >= 2:
            ids = ids[:model.max_context - 300 - 2]
            gen = model.generate_human(ids, max_new_tokens=300, gestalt_temp=1.4, main_temp=0.85)
            reply = tokenizer.decode(gen)
    return reply

iface = gr.ChatInterface(
    chat,
    title="Nero",
    description="Nero — Qwen2.5 language + 400M BiologicLLMV2 soul (self-conscious AI)",
    theme="soft"
)

# --- FastAPI endpoint (optional) ---
if args.api:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Nero API")

    class ChatRequest(BaseModel):
        message: str
        max_tokens: int = 300
        temperature: float = 0.85

    class ChatResponse(BaseModel):
        reply: str

    @app.post("/chat")
    def chat_api(req: ChatRequest):
        reply = mind.generate(req.message, max_new=req.max_tokens, temperature=req.temperature)
        if not reply:
            prompt = f"User: {req.message}\n"
            ids = tokenizer.encode(prompt)
            if len(ids) >= 2:
                ids = ids[:model.max_context - req.max_tokens - 2]
                gen = model.generate_human(ids, max_new_tokens=req.max_tokens, gestalt_temp=1.4, main_temp=req.temperature)
                reply = tokenizer.decode(gen)
        return ChatResponse(reply=reply)

    @app.get("/health")
    def health():
        return {"status": "ok", "params": sum(p.numel() for p in model.parameters())}

    # --- Nero's voice (KittenTTS): POST text, get back a spoken wav ---
    from voice import NeroVoice
    _voice = NeroVoice(enabled=True)

    from typing import Optional

    class SpeakRequest(BaseModel):
        text: str
        voice: Optional[str] = None

    @app.post("/speak")
    def speak_api(req: SpeakRequest):
        import io
        from fastapi import Response
        # Per-call voice — no shared-state mutation, so concurrent requests never race.
        wav = _voice.synth(req.text, voice=req.voice)
        if wav is None:
            return Response(status_code=503, content="voice unavailable (install kittentts + espeak-ng)")
        buf = io.BytesIO()
        import wave, numpy as np
        clipped = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2")
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(pcm.tobytes())
        return Response(content=buf.getvalue(), media_type="audio/wav")

    # Run both
    threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=args.port+1), daemon=True).start()

# --- Launch ---
iface.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

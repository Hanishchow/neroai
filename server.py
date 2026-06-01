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
    model.load_state_dict(sd)
    print(f"Loaded: {args.load}")

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
    description="200M param self-conscious AI",
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

    # Run both
    threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=args.port+1), daemon=True).start()

# --- Launch ---
iface.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
